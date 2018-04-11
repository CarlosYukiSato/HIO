# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:56:18 2017

@author: carlos.sato, giovanni.baraldi
"""
###############################################################################

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import FindCenter
from common import printw

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import skcuda
from pycuda.gpuarray import if_positive as gpuif
from pycuda.gpuarray import sum as gpusum
from time import time
import scipy.ndimage

Nstart = 16		# Run the algorithm Nstart to Nend times, Run# is a seed and will be put on the filename
Nend = 17
names = ['Pinhole2']	# Reconstruction for those names
iterations = 5000
bFORCE_REAL = False	# Set to true if the rspace is real.
IM_HALFSIZE = 480	# half the image size to use. Our data is limited to 960x960.
IM_CENTERY = 498	# XY Center of the image.
IM_CENTERX = 683
PerfLevel = 10		# The error will be registered every X iterations. Higher number means better performance.
bUsePinholeMask = False	# True = Load the pinhole mask. False = Use a msize X msize mask.
msize = 100		# size of the mask to be used.

Run = 0
beta = 0.9

from pycuda.elementwise import ElementwiseKernel
CuCopy = ElementwiseKernel(
"pycuda::complex<float> *x, pycuda::complex<float> *y",
"""
x[i] = y[i];
""",
"CuCopy",
preamble="#include <pycuda-complex.hpp>",)	# I dont trust python's assignment operator

ApplyDifPad = ElementwiseKernel(
"pycuda::complex<float> *kspace, float *difpad",
"""
	if(difpad[i] >= 0)
	{
		float ksabs = abs(kspace[i]) + 1E-10;
		kspace[i] *= difpad[i]/ksabs;
	}
""",
"ApplyDifPad",
preamble="#include <pycuda-complex.hpp>",)

CuHIO = ElementwiseKernel(
"pycuda::complex<float> *rspace, pycuda::complex<float> *kspace, pycuda::complex<float> *buffer, pycuda::complex<float> *sample, bool *Mask", 
"""
	if(Mask[i])
		sample[i] = rspace[i];

	rspace[i] = buffer[i] - 0.9f*rspace[i];

	if(sample[i].real() < 0)
		sample[i] = rspace[i];

	if(Mask[i])
		rspace[i] = sample[i];

	buffer[i] = rspace[i];
""",
"CuHIO",
preamble="#include <pycuda-complex.hpp>",)

CuError = ElementwiseKernel(
"pycuda::complex<float> *ktemp, float* difpad, float* kerror", 
"""
	if(difpad[i] >= 0)
	{
		float errabs = abs(ktemp[i]) - difpad[i];
		kerror[i] = abs(errabs);
	}
""",
"CuError",
preamble="#include <pycuda-complex.hpp>",)

def SaveImage(fname, Image):
	phases = np.angle(Image).astype(np.float32)	# Save amplitude and phase in different files
	absol = np.absolute(Image).astype(np.float32)
	phases[absol==0] = 0
	np.save(fname + '_amp_hio', absol)
	np.save(fname + '_phase_hio', phases)

	maxv = absol.max()
	Normalized = (255*np.sqrt((1.0/maxv)*absol)).astype(np.uint8)
	cv2.imwrite(fname+'_hio.png', Normalized)
	
def HIO(HDR_FILE_PATH, SAVE_FILES_PATH, Seed):
	curtime = time()
	np.random.seed(Seed)

	DifPad_cpu = np.load(HDR_FILE_PATH + '.npy')
	DifPad_cpu = DifPad_cpu[IM_CENTERY-IM_HALFSIZE:IM_CENTERY+IM_HALFSIZE,IM_CENTERX-IM_HALFSIZE:IM_CENTERX+IM_HALFSIZE]

	Negative_Mask = DifPad_cpu == -1
	Mask = np.zeros(DifPad_cpu.shape,dtype=np.bool)

	if bUsePinholeMask:
		PinholeMask = np.asarray(cv2.imread("PinholeMask.png",cv2.IMREAD_GRAYSCALE))
		maxphmask = np.max(PinholeMask)
		Mask = PinholeMask > 0.5*maxphmask
	else:
		Mask[:,:] = False
		qsize = msize//2
		Mask[IM_HALFSIZE-qsize:IM_HALFSIZE+qsize,IM_HALFSIZE-qsize:IM_HALFSIZE+qsize] = True;
	
	# start at random phases
	phase_angle = np.random.rand(DifPad_cpu.shape[0],DifPad_cpu.shape[1]).astype(np.float32)*2.0*np.pi

	#shift K00 for FFT
	DifPad_cpu = np.fft.ifftshift(DifPad_cpu).astype(np.float32)
	Negative_Mask = np.fft.ifftshift(Negative_Mask).astype(np.bool)

	DifPad_cpu[Negative_Mask] = 0
	k_space_cpu = DifPad_cpu * np.exp(1j*phase_angle)
	difpadsum = np.sum(DifPad_cpu)
	DifPad_cpu[Negative_Mask] = -1

	DifPad = gpuarray.to_gpu(DifPad_cpu)
	k_space = gpuarray.to_gpu(k_space_cpu)
	MaskBoolean = gpuarray.to_gpu(Mask)

	plan_forward = cu_fft.Plan(DifPad.shape, np.complex64, np.complex64)
	plan_inverse = cu_fft.Plan(DifPad.shape, np.complex64, np.complex64)
	ZeroVector32f = gpuarray.zeros(DifPad.shape, np.float32)
	ZeroVector64c = gpuarray.zeros(DifPad.shape, np.complex64)

	r_space = gpuarray.zeros(DifPad.shape, np.complex64)
	buffer_r_space = gpuarray.zeros(DifPad.shape, np.complex64)
	sample = gpuarray.zeros(k_space.shape, np.complex64) 
	R2D = gpuarray.zeros(DifPad_cpu.shape,dtype=np.complex64)
	ktemp = gpuarray.zeros(DifPad_cpu.shape,np.complex64)
	errorkspace = gpuarray.zeros(DifPad_cpu.shape,np.float32)

	cu_fft.ifft(k_space, buffer_r_space, plan_inverse, True)
	if bFORCE_REAL:
		buffer_r_space = buffer_r_space.__abs__().astype(np.complex64)
	
	RfacF = []
	errorcpu=1
	iter = 1
	store = 0
	toperr = 1E15

	print('Init time: ' + str(int(1000*(time()-curtime))) + 'ms')
	curtime = time()

	while (iter < iterations):

		CuHIO(r_space, k_space, buffer_r_space, sample, MaskBoolean)

		cu_fft.fft(r_space, k_space, plan_forward) 
		#k_space[0,0] = k_space[0,0].__abs__().astype(np.complex64) # Set average rspace phase to 0

		ApplyDifPad(k_space,DifPad)	# Replace kspace aplitudes by the measured ones

		cu_fft.ifft(k_space, r_space, plan_inverse, True)

	    	#### ERRORS ####

		if iter%PerfLevel==0:
			cu_fft.fft(sample,ktemp,plan_forward)
			CuError(ktemp,DifPad,errorkspace)
			errorF = gpusum(errorkspace)
			errorcpu = errorF.get()/difpadsum

	    		RfacF.append(errorcpu + 0)

	    		# Find the iteration with the best error
	    		if errorcpu <= toperr:
				toperr = errorcpu + 0
				CuCopy(R2D,r_space)
		else:
			RfacF.append(errorcpu + 0)

		if bFORCE_REAL:
			r_space = r_space.real.astype(np.complex64)
		iter = iter + 1

		if iter%100 == 0:
			printw('Iteration: ' + str(iter))

	deltaT = time()-curtime
	print('')
	print('Run in: ' + str(int(deltaT)) + 's, at ' + str(int(iterations/deltaT)) + 'ops/s')
	print(toperr)

	try_code = 0 	# 0 = first attempt; 1 = second attempt; 2 = waiting for prompt
			# 3 = aborted; 4 = saved to location

	while try_code < 3: # Save reconstruction. If the desired file path gives us an error, try saving in other places.
		try:
			if try_code == 1: # If second attemp, try saving at current folder
				SAVE_FILES_PATH = 'OSS_Files/Temp/' + name + '_Run' + str(Run) + '_'
			elif try_code == 2:
				SAVE_FILES_PATH = raw_input("Path to save files.\n") # If cant, prompt user
				if SAVE_FILES_PATH == '0':
					try_code = 3
					raise

			cu_fft.fft(R2D, k_space, plan_forward) # Get best result
				
			SaveImage(SAVE_FILES_PATH + "rspace", R2D.get())
			SaveImage(SAVE_FILES_PATH + "kspace", k_space.get())

			plt.subplot(1,1,1)
			plt.plot(np.asarray(RfacF,dtype=np.float32))
			plt.ylabel('Fourier R-factor')
			plt.xlabel('Iteration')
			plt.tight_layout()
			plt.savefig(SAVE_FILES_PATH + "ErrorHIO.png",dpi=50)
			plt.close()
			try_code = 4
		except:
			if try_code == 0:
				try_code = 1
			elif try_code == 1:
				try_code = 2

			if try_code != 3:
				print('Couldnt write to ' + SAVE_FILES_PATH)

for name in names:
	for Run in range(Nstart,Nend):
		print(name + ': Run ' + str(Run))
		HIO('HDR/' + name, 'OSS_Files/' + name + '/Run' + str(Run) + '_', (Run*24759+285)%5897 )

