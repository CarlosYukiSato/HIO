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

iterations = 50000
IM_HALFSIZE = 480	# half the image size to use. Our data is limited to 960x960.
IM_CENTERY = 498	# XY Center of the image.
IM_CENTERX = 683
PerfLevel = 2		# The error will be registered every X iterations. Higher number means better performance.
msize = 104		# size of the mask to be used.

Run = 0
beta = 0.9

from pycuda.elementwise import ElementwiseKernel

MakeDescent = ElementwiseKernel(
"pycuda::complex<float> *outsidefft, pycuda::complex<float> *kspace",
"""
	pycuda::complex<float> mul = kspace[i]*conj(outsidefft[i]);
	float dp = 1E-15*mul.imag();
	kspace[i] *= pycuda::complex<float>(cos(dp),sin(dp));
""",
"MakeDescent",
preamble="#include <pycuda-complex.hpp>",)

Error = ElementwiseKernel(
"float* err, pycuda::complex<float> *rspace", 
"""
	err[i] = abs(rspace[i]*conj(rspace[i]));
""",
"Error",
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
	
def Grad(HDR_FILE_PATH, SAVE_FILES_PATH, Seed):
	curtime = time()
	np.random.seed(Seed)
	
	rspacelena = np.zeros((512,512),dtype=np.complex64)
	phase_angle = np.random.rand(rspacelena.shape[0],rspacelena.shape[1]).astype(np.float32) - 0.5

	Lena = np.asarray(cv2.imread("Lena.png",cv2.IMREAD_GRAYSCALE),dtype=np.float32)[64:192,64:192]
	rspacelena[128:256,128:256] = Lena[:,:]
	InvMask = rspacelena == 0

	rspace = gpuarray.to_gpu(rspacelena)
	kspace = gpuarray.zeros(rspace.shape,dtype=np.complex64)
	outsidefft = gpuarray.zeros(rspace.shape,dtype=np.complex64)
	InverseMask = gpuarray.to_gpu(InvMask)

	plan_forward = cu_fft.Plan(rspace.shape, np.complex64, np.complex64)
	plan_inverse = cu_fft.Plan(rspace.shape, np.complex64, np.complex64)
	ZeroVector32f = gpuarray.zeros(rspace.shape, np.float32)
	ZeroVector64c = gpuarray.zeros(rspace.shape, np.complex64)

	cu_fft.fft(rspace,kspace,plan_forward)
	kspace *= gpuarray.to_gpu(np.exp(1j*phase_angle))

	errorR = gpuarray.zeros(rspace.shape,np.float32)
	RfacF = []
	iter = 1
	toperr = 1E15

	print('Init time: ' + str(int(1000*(time()-curtime))) + 'ms')
	curtime = time()

	normal = np.absolute(rspacelena).sum()

	while (iter < iterations):
		cu_fft.ifft(kspace,rspace,plan_inverse,True)
		rspace = gpuif(InverseMask,rspace,ZeroVector64c)
		cu_fft.fft(rspace,outsidefft,plan_forward)
		MakeDescent(outsidefft,kspace)
		Error(errorR,rspace)
		errorF = gpusum(errorR).get()/normal
		if iter%1000 == 0:
			print(errorF)

		iter = iter + 1

	deltaT = time()-curtime
	print('')
	print('Run in: ' + str(int(deltaT)) + 's, at ' + str(int(iterations/deltaT)) + 'ops/s')
	print(toperr)
	
				
	SaveImage(SAVE_FILES_PATH + "rspace", rspace.get())

	plt.subplot(1,1,1)
	plt.plot(np.asarray(RfacF,dtype=np.float32))
	plt.ylabel('Fourier R-factor')
	plt.xlabel('Iteration')
	plt.tight_layout()
	plt.savefig(SAVE_FILES_PATH + "ErrorGrad.png",dpi=50)
	plt.close()

Grad("Grad","Grad",2)

