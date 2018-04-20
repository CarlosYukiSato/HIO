# giovanni.baraldi


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from common import printw

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import skcuda
from pycuda.gpuarray import if_positive as gpuif
from pycuda.gpuarray import sum as gpusum
from time import time
import scipy.ndimage

iterations = 100000
IM_HALFSIZE = 480	# half the image size to use. Our data is limited to 960x960.
IM_CENTERY = 498	# XY Center of the image.
IM_CENTERX = 683
PerfLevel = 4		# The error will be registered every X iterations. Higher number means better performance.
msize = 104		# size of the mask to be used.

Run = 0
beta = 0.9

from pycuda.elementwise import ElementwiseKernel

MakeDescent = ElementwiseKernel(
"pycuda::complex<float> *outsidefft, pycuda::complex<float> *kspace, float* prevPhasegrad, float* prevAmpgrad, bool* NegMask, bool AmpOnly",
"""
	const float momentum = 0.9997f;

	pycuda::complex<float> grad = kspace[i]*conj(outsidefft[i])*1E-14f;

	float newgrad = prevPhasegrad[i]*momentum + grad.imag();

	if(!AmpOnly)
	{
		kspace[i] *= pycuda::complex<float>(cos(newgrad),sin(newgrad));
		prevPhasegrad[i] = newgrad;
	}

	if(NegMask[i])
	{
		newgrad = 1E4f*grad.real()/float(abs(kspace[i])+1E-10) + prevAmpgrad[i]*0.97f;
	
		kspace[i] -= kspace[i]*newgrad;
		prevAmpgrad[i] = newgrad;
	}

""",
"MakeDescent",
preamble="#include <pycuda-complex.hpp>",)

Error = ElementwiseKernel(
"float* err, pycuda::complex<float> *rspace", 
"""
	err[i] = (rspace[i]*conj(rspace[i])).real();
""",
"Error",
preamble="#include <pycuda-complex.hpp>",)

ApplyMask = ElementwiseKernel(
"pycuda::complex<float> *rspace, bool* Mask", 
"""
	if(Mask[i])
		rspace[i] = 0;
""",
"ApplyMask",
preamble="#include <pycuda-complex.hpp>",)


def SaveImage(fname, Image):
	phases = np.angle(Image).astype(np.float32)	# Save amplitude and phase in different files
	absol = np.absolute(Image).astype(np.float32)
	phases[absol==0] = 0
	np.save(fname + '_amp_grad', absol)
	np.save(fname + '_phase_grad', phases)

	maxv = absol.max()
	Normalized = (255*np.sqrt((1.0/maxv)*absol)).astype(np.uint8)
	cv2.imwrite(fname+'_grad.png', Normalized)
	
def CalcError(rspace,errorR,normal):
	Error(errorR,rspace)
	errorF = gpusum(errorR).get()/normal
	errorF = np.sqrt(errorF)
	return errorF

def Grad(HDR_FILE_PATH, SAVE_FILES_PATH, Seed):
	curtime = time()
	np.random.seed(Seed)
	
	rspacelena = np.zeros((512,512),dtype=np.complex64)
	phase_angle = np.random.rand(rspacelena.shape[0],rspacelena.shape[1]).astype(np.float32)*3 - 1.5

	Lena = np.asarray(cv2.imread("Lena.png",cv2.IMREAD_GRAYSCALE),dtype=np.float32)[64:192,64:192]

	for j in range(-64,64):
		for i in range(-64,64):
			Lena[j+64,i+64] = Lena[j+64,i+64]*np.exp(-(i*i+j*j)/500.0 )

	rspacelena[128:256,128:256] = Lena[:,:]
	MaskSupport = rspacelena > 0
	#MaskSupport[100:300,100:300] = True

	rspace = gpuarray.to_gpu(rspacelena)
	kspace = gpuarray.zeros(rspace.shape,dtype=np.complex64)
	gradPhase = gpuarray.zeros(rspace.shape,dtype=np.float32)
	gradAmp = gpuarray.zeros(rspace.shape,dtype=np.float32)
	outsidefft = gpuarray.zeros(rspace.shape,dtype=np.complex64)
	MaskSupport = gpuarray.to_gpu(MaskSupport)

	plan_forward = cu_fft.Plan(rspace.shape, np.complex64, np.complex64)
	plan_inverse = cu_fft.Plan(rspace.shape, np.complex64, np.complex64)

	cu_fft.fft(rspace,kspace,plan_forward)
	kspace *= gpuarray.to_gpu(np.exp(1j*phase_angle))

	kspace_cpu = np.fft.fftshift(kspace.get())
	NegMask_cpu = np.zeros(kspace.shape,dtype=np.bool);

	plt.subplot(3,2,5)
	plt.imshow(np.absolute(kspace_cpu[200:300,200:300]))

	#NegMask_cpu[257-3:257+2,257-3:257+2] = True
	kspace_cpu[NegMask_cpu] = kspace_cpu[NegMask_cpu]*1E6/(kspace_cpu[NegMask_cpu].__abs__()+1.0);

	plt.subplot(3,2,6)
	plt.imshow(np.absolute(kspace_cpu[200:300,200:300]))

	NegMask = gpuarray.to_gpu(np.fft.ifftshift(NegMask_cpu))
	kspace = gpuarray.to_gpu(np.fft.ifftshift(kspace_cpu))	

	errorF = 0.0
	errorR = gpuarray.zeros(rspace.shape,np.float32)
	RfacF = []
	iter = 1
	toperr = 1E15

	plt.subplot(3,2,1)
	plt.imshow(Lena)

	cu_fft.ifft(kspace,rspace,plan_inverse,True)
	plt.subplot(3,2,2)
	plt.imshow(rspace.real.get()[128:256,128:256])

	normal = np.sum(Lena*Lena)
	ApplyMask(rspace,MaskSupport)
	#RfacF.append(CalcError(rspace,errorR,normal))

	print('Init time: ' + str(int(1000*(time()-curtime))) + 'ms')
	curtime = time()

	while (iter < iterations):
		cu_fft.ifft(kspace,rspace,plan_inverse,True)
		ApplyMask(rspace,MaskSupport)
		cu_fft.fft(rspace,outsidefft,plan_forward)

		if iter < iterations/5:
			MakeDescent(outsidefft,kspace,gradPhase,gradAmp,NegMask,True)
		else:
			MakeDescent(outsidefft,kspace,gradPhase,gradAmp,NegMask,False)

		if iter%PerfLevel == 0:
			errorF = 100.0*CalcError(rspace,errorR,normal)
			RfacF.append(errorF)
			toperr = min(toperr,errorF)
		if iter%1000 == 0:
			printw('Iter: ' + str(iter) + ' \tError: ' + str(int(errorF)) + '.' + str(int(10*errorF)%10) + str(int(100*errorF)%10) + ' %')

		iter = iter + 1

	deltaT = time()-curtime
	print('')
	print('Run in: ' + str(int(deltaT)) + 's, at ' + str(int(iterations/deltaT)) + 'ops/s')
	print 'Best:', toperr
	
	cu_fft.ifft(kspace,rspace,plan_inverse,True)

	plt.subplot(3,2,3)
	plt.imshow(rspace.__abs__().get()[128:256,128:256])

	plt.subplot(3,2,4)
	plt.plot(np.asarray(RfacF,dtype=np.float32))
	plt.ylabel('Fourier R-factor')
	plt.xlabel('Iteration')
	plt.tight_layout()
	plt.show()
	#plt.savefig(SAVE_FILES_PATH + "ErrorGrad.png",dpi=50)
	plt.close()

Grad("Grad","Grad",2)

