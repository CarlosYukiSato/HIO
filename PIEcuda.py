# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:27:34 2018

@author: sato, giovanni.baraldi


Implementation of the PIE algorithm for solving ptychografic phase problem experiments
"""

PerfLevel = 2 # update error and probemax every X iterations, higher number = better performance.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ptychoTools
import common

from numpy.fft import fftn, ifftn
from common import printw

import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import skcuda
from pycuda.gpuarray import if_positive as gpuif
from pycuda.gpuarray import sum as gpusum
from pycuda.gpuarray import max as gpumax

from PIL import Image
im = Image.open('Lena.tiff')

from time import time
    
Lena = np.array(im)
del(im)
Lena =0.2989 * Lena[:,:,0] + 0.5870 * Lena[:,:,1] + 0.1140 * Lena[:,:,2]

im = Image.open('Lena.tiff')
BoatLake =  np.array(im)
del(im)
BoatLake =0.2989 * BoatLake[:,:,0] + 0.5870 * BoatLake[:,:,1] + 0.1140 * BoatLake[:,:,2]
BoatLake = BoatLake - np.min(BoatLake)
BoatLake *= BoatLake
BoatLake = BoatLake / np.max(BoatLake) - 0.5

finalModel = Lena * np.exp(1j * BoatLake * np.pi)

apertureDiameter = 50
overSampling = 4

imagApert = np.int(apertureDiameter * overSampling)
imagApert = [imagApert,imagApert]
aperture_cpu = ptychoTools.makeCircAperture(apertureDiameter,imagApert)

step = apertureDiameter * .5
positions = ptychoTools.makePositions(-2,2,5,step, 2)
positions = np.array(positions,dtype = np.int)

DifPads = ptychoTools.calcDifPad(finalModel,aperture_cpu,positions)

finalModel = finalModel[finalModel.shape[0]/2 - 150 : finalModel.shape[0]/2 + 150 , finalModel.shape[1]/2 - 150 : finalModel.shape[1]/2 + 150]


ampKSpace_cpu = ptychoTools.ptychoFFTShift(DifPads)

finalObj_cpu = ptychoTools.createFinalObj(DifPads,positions)
nApert = len(DifPads) #number of apertures from the number of positions it assumes
roiSize = (len(DifPads),np.shape(DifPads[0])[0],np.shape(DifPads[0])[1])

roiList = ptychoTools.listROI(positions,roiSize,finalObj_cpu.shape)

iterations = 200

betaObj = .9
betaApert = .9

errorF = np.zeros([int(iterations) , nApert])
lastError = np.ones([nApert])

iter = 1

rspaceShape = (np.shape(DifPads[0])[0],np.shape(DifPads[0])[1])

finalObj = gpuarray.to_gpu(finalObj_cpu)
rspace = gpuarray.zeros(rspaceShape,dtype=np.complex64)
exitWave = gpuarray.zeros(rspaceShape,dtype=np.complex64)
buffer_exitWave = gpuarray.zeros(rspaceShape,dtype=np.complex64)
kspace = gpuarray.zeros(rspaceShape,dtype=np.complex64)
k_spaceAmp = gpuarray.zeros(rspaceShape,dtype=np.float32)
aperture = gpuarray.to_gpu(aperture_cpu)

plan_forward = cu_fft.Plan(rspaceShape, np.complex64, np.complex64)
plan_inverse = cu_fft.Plan(rspaceShape, np.complex64, np.complex64)

ampKSpace = []
for k in range(0,len(ampKSpace_cpu)):
	ampKSpace.append( gpuarray.to_gpu(ampKSpace_cpu[k]) )

from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
SourceCopyFromFinal = SourceModule("""
#include <pycuda-complex.hpp>
__global__ void CopyFromFinal(pycuda::complex<float> *dest, pycuda::complex<float> *src, 
				int ofy, int ofx, int roisizex, int objsizex, int size)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{
		int j = int(index/roisizex);
		int i = index - j*roisizex;
		
		j += ofy;
		i += ofx;

		int largeptr = j*objsizex + i;
		
		dest[index] = src[largeptr];
		index += numthreads;
	}
}
""")

SourceCopyToFinal = SourceModule("""
#include <pycuda-complex.hpp>
__global__ void CopyToFinal(pycuda::complex<float> *finalObj, pycuda::complex<float> *rspace, pycuda::complex<float> *exitwave,
				pycuda::complex<float> *buffer_exitwave, pycuda::complex<float> *aperture, float updateFactorObj,
				int ofy, int ofx, int roisizex, int objsizex, int size)
{
	int numthreads = blockDim.x*gridDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	while(index<size)
	{
		pycuda::complex<float> result = exitwave[index] - buffer_exitwave[index];
		result *= conj(aperture[index])*updateFactorObj;
		result += rspace[index];

		int j = int(index/roisizex);
		int i = index - j*roisizex;

		j += ofy;
		i += ofx;
		
		finalObj[j*objsizex + i] = result;
		index += numthreads;
	}
}
""")


CopyFromFinal = SourceCopyFromFinal.get_function("CopyFromFinal")
CopyToFinal = SourceCopyToFinal.get_function("CopyToFinal")

ApplyDifPad = ElementwiseKernel(
"pycuda::complex<float> *kspace, float *difpad, float* kamp",
"""
	if(difpad[i] >= 0)
	{
		float ksabs = abs(kspace[i]) + 1E-15;
		kspace[i] *= difpad[i]/ksabs;
		kamp[i] = abs(ksabs-difpad[i]);
	}
""",
"ApplyDifPadWithError",
preamble="#include <pycuda-complex.hpp>",)

ExitwaveAndBuffer = ElementwiseKernel(
"pycuda::complex<float> *exitwave,pycuda::complex<float> *buffer_exitwave, pycuda::complex<float>* aperture, pycuda::complex<float>* rspace",
"""
	pycuda::complex<float> res = rspace[i]*aperture[i];
	exitwave[i] = res;
	buffer_exitwave[i] = res;
""",
"ExitwaveAndBuffer",
preamble="#include <pycuda-complex.hpp>",)

curtime = time()

difpadsums = []
for aper in range(nApert):
	difpadsums.append( gpusum(ampKSpace[aper]).get() )

apertMaxSqrt = 1.0

while (iter < iterations):
	apertIndex = np.random.permutation(nApert)
    
	for cont in range(nApert):
		aper =  apertIndex[cont]

		offsety = np.int32(roiList[aper][0])
		offsetx = np.int32(roiList[aper][2])
		roisizex = np.int32(roiList[aper][3]-roiList[aper][2])
		roisizey = np.int32(roiList[aper][1]-roiList[aper][0])
		objsizex = np.int32(finalObj_cpu.shape[1])

		roisize = roisizex*roisizey

		CopyFromFinal(rspace, finalObj, offsety, offsetx, roisizex, objsizex, roisize, grid=(1,1), block=(512,1,1))

		if (iter-1)%PerfLevel == 0:
			apertMaxSqrt = gpumax(aperture.__abs__()).get()**2
        
		ExitwaveAndBuffer(exitWave, buffer_exitWave, aperture, rspace)
		
		cu_fft.fft(exitWave,kspace,plan_forward)

        	ApplyDifPad(kspace,ampKSpace[aper],k_spaceAmp)
        
		cu_fft.ifft(kspace,exitWave,plan_inverse,True)
		updateFactorObj = np.float32(betaObj/apertMaxSqrt)

		CopyToFinal(finalObj, rspace, exitWave, buffer_exitWave, aperture, updateFactorObj, offsety, offsetx, roisizex, objsizex, roisize, grid=(1,1), block=(512,1,1))
        
		if (iter-1)%PerfLevel == 0:
			lastError[aper] = gpusum(k_spaceAmp).get()/difpadsums[aper]
		errorF[int(iter-1),aper] = lastError[aper]
      
	iter = iter + 1
	if iter%10 == 0:
		print(iter)

curtime = time()-curtime
print "Done in " + str(int(curtime)) + "." + str(int(curtime*10)%10) + "s"

plt.plot(errorF)

plt.subplot(2, 2, 1)
imgplot = plt.imshow(np.absolute(finalObj.get()))

RGB = common.CMakeRGB(finalObj.get())
plt.subplot(2, 2, 2)
imgplot = plt.imshow(RGB)
plt.subplot(2, 2, 3)
imgplot = plt.imshow(np.absolute(finalModel)) 

plt.subplot(2, 2, 4)
plt.plot(errorF)
plt.ylabel('Fourier R-factor')
plt.xlabel('Iteration')
#plt.title('Worst R-factor: %s' %(errorF*100))
plt.tight_layout()
plt.show()

