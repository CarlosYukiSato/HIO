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
from common import printw

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import skcuda
import Kernels
from pycuda.gpuarray import if_positive as gpuif
from pycuda.gpuarray import sum as gpusum
from time import time
import scipy.ndimage

Nstart = 0		# Run the algorithm Nstart to Nend times, Run# is a seed and will be put on the filename
Nend = 1

names = ['Pinhole2']	# Reconstruction for those names
iterations = 500
bFORCE_REAL = False	# Set to true if the rspace is real.
IM_HALFSIZE = 480	# half the image size to use. Our data is limited to 960x960 (480).
PerfLevel = 3		# The error will be registered every X iterations. Higher number means better performance.
bUsePinholeMask = False	# True = Load the pinhole mask. False = Use a msize X msize mask.
bUseCircularMask = True
msize = 98		# size of the mask to be used.
Radius = 49

Run = 0
beta = 0.9

Kernels.LoadKernel("Scripts/HIOKernels.cu", np.int32(4*IM_HALFSIZE*IM_HALFSIZE))

ApplyDifPad = Kernels.GetFunction("ApplyDifPad")
HIOStep = Kernels.GetFunction("HIOStep")
Copy = Kernels.GetFunction("Copy")
Error = Kernels.GetFunction("Error1DifCF")

def SaveImage(fname, Image):
	np.save(fname + '_hio', Image)

	absol = np.absolute(Image).astype(np.float32)
	maxv = absol.max()
	Normalized = (255*np.sqrt((1.0/maxv)*absol)).astype(np.uint8)
	cv2.imwrite(fname+'_hio.png', Normalized)
	
def HIO(HDR_FILE_PATH, SAVE_FILES_PATH, Seed):
	global bFORCE_REAL
	curtime = time()
	np.random.seed(Seed)

	DifPad_cpu = np.load(HDR_FILE_PATH + '.npy')
	IndexMax = np.argmax(DifPad_cpu)
	IM_CENTERY = IndexMax//DifPad_cpu.shape[1]
	IM_CENTERX = IndexMax - IM_CENTERY*DifPad_cpu.shape[1]
	DifPad_cpu = DifPad_cpu[IM_CENTERY-IM_HALFSIZE:IM_CENTERY+IM_HALFSIZE,IM_CENTERX-IM_HALFSIZE:IM_CENTERX+IM_HALFSIZE]

	Negative_Mask = DifPad_cpu == -1

	Mask = np.zeros(DifPad_cpu.shape,dtype=np.bool)

	if bUsePinholeMask:
		PinholeMask = np.asarray(cv2.imread("PinholeMask.png",cv2.IMREAD_GRAYSCALE))
		maxphmask = np.max(PinholeMask)
		Mask = PinholeMask > 0.5*maxphmask
	elif bUseCircularMask:
		Mask[:,:] = False
		for j in range(-Radius,+Radius+1):
			for i in range(-Radius,+Radius+1):
				if j**2 + i**2 < Radius**2:
					Mask[IM_HALFSIZE+j,IM_HALFSIZE+i] = True
	else:	
		Mask[:,:] = False
		qsize = msize//2
		Mask[IM_HALFSIZE-qsize:IM_HALFSIZE+qsize,IM_HALFSIZE-qsize:IM_HALFSIZE+qsize] = True;
	
	# start at random phases
	phase_angle = np.random.rand(DifPad_cpu.shape[0],DifPad_cpu.shape[1]).astype(np.float32)*2.0*np.pi

	#shift K00 for FFT
	DifPad_cpu = np.fft.ifftshift(DifPad_cpu).astype(np.float32)
	Negative_Mask = np.fft.ifftshift(Negative_Mask).astype(np.bool)

	plan_forward = cu_fft.Plan(DifPad_cpu.shape, np.complex64, np.complex64)
	plan_inverse = cu_fft.Plan(DifPad_cpu.shape, np.complex64, np.complex64)

	DifPad_cpu[Negative_Mask] = 0

	k_space_cpu = DifPad_cpu * np.exp(1j*phase_angle)

	difpadsum = np.sum(DifPad_cpu)
	DifPad_cpu[Negative_Mask] = -1

	DifPad = gpuarray.to_gpu(DifPad_cpu)
	k_space = gpuarray.to_gpu(k_space_cpu)
	MaskBoolean = gpuarray.to_gpu(Mask)

	r_space = gpuarray.zeros(DifPad.shape, np.complex64)
	buffer_r_space = gpuarray.zeros(DifPad.shape, np.complex64)
	sample = gpuarray.zeros(k_space.shape, np.complex64) 
	R2D = gpuarray.zeros(DifPad_cpu.shape,dtype=np.complex64)
	ktemp = gpuarray.zeros(DifPad_cpu.shape,dtype=np.complex64)
	errorkspace = gpuarray.zeros((Kernels.numthreads),np.float32)

	cu_fft.ifft(k_space, buffer_r_space, plan_inverse, True)
	if bFORCE_REAL:
		buffer_r_space = buffer_r_space.__abs__().astype(np.complex64)
	
	RfacF = []
	errorF=1
	iter = 1
	toperr = 1E15

	print SAVE_FILES_PATH, 'init time:', int(1000*(time()-curtime)), 'ms'
	curtime = time()

	while (iter < iterations):
		#if iter == iterations//2:
		#	bFORCE_REAL = False

		HIOStep(r_space, k_space, buffer_r_space, sample, MaskBoolean)

		cu_fft.fft(r_space, k_space, plan_forward)
		#k_space[0,0] = k_space[0,0].__abs__().astype(np.complex64) # Set average rspace phase to 0

		ApplyDifPad(k_space,DifPad)	# Replace kspace aplitudes by the measured ones

		cu_fft.ifft(k_space, r_space, plan_inverse, True)

	    	#### ERRORS ####

		if iter%PerfLevel==0:
			cu_fft.fft(sample, ktemp, plan_forward)
			Error(ktemp,DifPad,errorkspace)
			errorF = errorkspace.get().sum()/difpadsum

	    		RfacF.append(errorF + 0)

	    		# Find the iteration with the best error
	    		if errorF <= toperr:
				toperr = errorF + 0
				Copy(R2D,r_space)
				
		else:
			RfacF.append(errorF + 0)

		if bFORCE_REAL:
			r_space = r_space.real.astype(np.complex64)
		iter = iter + 1

		if iter%100 == 0:
			printw('Iteration: ' + str(iter))

	cu_fft.fft(R2D, k_space, plan_forward) # Get best result

	deltaT = time()-curtime
	printw('Run in: ' + str(int(deltaT)) + 's, at ' + str(int(iterations/deltaT)) + 'ops/s with error ' + str(toperr))

	plt.imshow(np.absolute(R2D.get()))
	plt.show()

	try:
		strindex = -1
		for istr in range(len(SAVE_FILES_PATH)):
			if SAVE_FILES_PATH[istr] == '/':
				strindex = istr

		if strindex > 0:
			os.system('mkdir -p ' + SAVE_FILES_PATH[0:strindex])

		SaveImage(SAVE_FILES_PATH + "rspace", R2D.get())
		SaveImage(SAVE_FILES_PATH + "kspace", k_space.get())

		plt.subplot(1,1,1)
		plt.plot(np.asarray(RfacF,dtype=np.float32))
		plt.ylabel('Fourier R-factor')
		plt.xlabel('Iteration')
		plt.tight_layout()
		plt.savefig(SAVE_FILES_PATH + "ErrorHIO.png",dpi=50)
		plt.close()

	except:
		print 'Couldnt write to ' + SAVE_FILES_PATH

	print('')

for name in names:
	HIO(name, 'RecFiles/' + name + '_', (Run*24759+285)%5897 )

