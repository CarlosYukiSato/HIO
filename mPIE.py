import numpy as np
from ptychohelper import *
# -*- coding: utf-8 -*-
""" @author: sato, giovanni.baraldi
Implementation of the (e/m/r)PIE algorithm for solving ptychografic phase problem experiments """

np.random.seed(1)
iPosErrorAmount = 0
fNoiseLevel = 1.0	# Simulated noise
iPosCorrection = 0 	# Try to improve ROI positioning by given amount. 0 = Disabled. 1 = recommended.

NumModes = 1 		# Number of probe incoherent modes
bInteractive = True 	# A pyplot window will be updated every 50 iterations.
bPhaseShift = False 	# Subpixel shift is considered on each ROI. Mildly expensive.
PerfLevel = 3 		# update error and probemax every X iterations, higher number = better performance.
			# For some wierd reason, can improve results (?)

bMakeReset = False 	# Object will be reset to noise every 'resetEvery' iterations.
bCropObject = True	# Dampen regions of the object not touched by the probe.
			# /\ Both the above should improve probe estimation.

bNormalizeEach = False 	# Normalize each probe independently.  	-> ||Pm|| = 1
bNormalizeAll = False 	# Normalize all probes as one. 		-> sum( ||Pm|| ) = 1
bOrthogonal = False 	# Impose probes are orthogonal to eachother.
bRelaxedOrtho = False 	# Impose probes are orthogonal to the first mode only.

momentum = 0.9		# < 0.9 for fast convergence, up to ~0.99 for very dificult problems ( requires lots of iterations )
iterations = 500	# Total number of iterations.
CropSav = 5/(1-momentum)# Dont crop object at the beginning and end of iterations by this amount.
crop_factor = 0.99	# Object dampen factor.
crop_threshold = 0.05	# Threshold for what 'touch' means. When the probe is normalized,
			# crop_threshold = 1 -> average ( |probe|**2 ) .
			
resetEvery = 100 	# Only if bMakeReset.
resetTo = 3*resetEvery 	# Last reset, from there on the probe is considered 'final'.

betaObj = 0.9		# Object update factor.
betaApert = 0.0		# Probe update factor. 0.0 ~ 0.1 depending on initial probe estimation.
sincguesses = [64] 	# List of initial sinc radius guesses for the probes.
realsincs = [64] 	# List of sinc radius for difpad calculation

# Size of the difpad/rspace
imagApert = np.int32(256) 	# Size of the difpads
imagApert = [imagApert,imagApert]

# GetRois(M,N,S,im) = AxB square grid of step size S and difpad size im
roiList = GetRois(11,11,23,imagApert) + GetRois(9,9,31,imagApert) # Two square grids will avoid artifacts.
#roiList = GetRois(17,17,37,imagApert) + GetRois(13,13,43,imagApert)
#roiList = GetRois(9,9,32,imagApert)

ApplyJitter(roiList,10,32) # offset ROI and apply random jitter to avoid artifacts

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import common

from numpy.fft import fftn, ifftn
from common import printw

import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
import skcuda
from copy import deepcopy
from ptychoposcorrection import *
from time import time
import Kernels

# Load CUDA kernel modules, params: name, number of pixels, list of values to replace on source code
Kernels.LoadKernel("Scripts/PIEKernels.cu", np.int32(imagApert[0]*imagApert[1]),["#define NumModes X", NumModes])

""" Get CUDA kernel functions """

# Basic ePIE kernels
ReplaceInObject = Kernels.GetFunction("ReplaceInObject")
ApplyDifPad = Kernels.GetFunction("ApplyDifPad")
ExitwaveAndBuffer = Kernels.GetFunction("ExitwaveAndBuffer")
ApertureAbs2 = Kernels.GetFunction("ApertureAbs2")
ObjectAbs2 = Kernels.GetFunction("ObjectAbs2")
CopyFromROI = Kernels.GetFunction("CopyFromROI")
CropObject = Kernels.GetFunction("CropObject")
UpdateProbeAndRspace = Kernels.GetFunction("UpdateProbeAndRspace")

# Only used if bPhaseShift = True
PhaseShiftFunc = Kernels.GetFunction("PhaseShift")

# Multiprobe
ModeSub21 = Kernels.GetFunction("ModeSub21")
ScalarProd = Kernels.GetFunction("ScalarProd")
ModeMultiply = Kernels.GetFunction("ModeMultiply")
NormSquared = Kernels.GetFunction("NormSquared")
ModeSquared = Kernels.GetFunction("ModeSquared")

finalModel = LoadExtendedLenaModel() # Get default(ish) simulated model. Replace this by your image.
#finalModel = LoadNISTModel() # Another option for a model.

# Probes to generate simulated diffraction image. Other commented options below.
# Circular probes are much easier to solve.
aperture_cpu = MakeSincApertures(realsincs,imagApert)
#aperture_cpu = MakeGaussApertures(>[sigmas]<,imagApert, optionals: >gaussian threshold<, >[amplitudes]<, >[rolls]<)
#aperture_cpu = MakeCircApertures(>[diameters]<,imagApert, optionals: >[amplitudes]<, >[rolls]<)

#aperture_cpu = [np.load('Medidas/probe0.npy')] # A propagated diffuser
#aperture_cpu[0] /= np.sqrt(np.sum(abs(aperture_cpu[0])**2)) # Normalize the previous probe

newroiList, errors = MakePosErrors(roiList,iPosErrorAmount) # Simulate position error by +-iPosErrorAmount
DifPads,difpadsums = GetDifPads(finalModel,aperture_cpu,newroiList,fNoiseLevel)

rspaceShape = (aperture_cpu[0].shape[0],aperture_cpu[0].shape[1])
gpuspaceShape = (NumModes*aperture_cpu[0].shape[0],aperture_cpu[0].shape[1])
nApert = np.int32(len(DifPads)) #number of apertures from the number of positions it assumes

errorF = np.zeros([int(iterations-1) , nApert])
lastError = np.ones([nApert]) # Cache in case PerfLevel > 1

ObjShape = MakeObjShapeFromROIS(roiList)
finalObj = GetRandomObj(ObjShape) # Start obj with noise
velocities = cgpuarray(ObjShape)
rspace = cgpuarray(rspaceShape)
fcachevector = fgpuarray(Kernels.numthreads)
ccachevector = cgpuarray(Kernels.numthreads)
dotvector = cgpuarray(Kernels.numthreads)

exitWave = cgpuarray(gpuspaceShape)
buffer_exitWave = cgpuarray(gpuspaceShape)
kspace = cgpuarray(gpuspaceShape)

cufftplan = cu_fft.Plan(rspaceShape, np.complex64, np.complex64, batch=NumModes, idist=rspaceShape[0]*rspaceShape[1], odist=rspaceShape[0]*rspaceShape[1])

apertMax = 1.0
ObjMax = farray(nApert)+1.0

# Replace below by the commented line for perfect probe estimation.
aperture = gpuarray.to_gpu(np.reshape(np.asarray(MakeSincApertures(sincguesses,imagApert)).astype(np.complex64),gpuspaceShape))
#aperture = gpuarray.to_gpu(np.reshape(np.asarray(aperture_cpu).astype(np.complex64),gpuspaceShape))

# Send rois for object cropping. You may want to resend them if position correction is enabled.
roioffsetsgpu = []
for roi in roiList:
	roioffsetsgpu.append( [roi[0],roi[2]] )
roioffsetsgpu = gpuarray.to_gpu( np.asarray(roioffsetsgpu).astype(np.int32) )

if bCropObject == False:
	finalObj *= 0.1 # Multiply by 0.1 for better image display
	
iter = 1
relative_iter = 1
curtime = time()

if bInteractive:
	plt.ion()
	plt.show()
		
while (iter < iterations):
	if bInteractive and iter%50 == 0:
		plt.subplot(1,2,1)
		cpuobj = finalObj.__abs__().astype(np.float32).get()
		#cpuobj[cpuobj>1] = 1
		plt.imshow(cpuobj)
		plt.subplot(1,2,2)
		plt.imshow(aperture.__abs__().astype(np.float32).get())
		plt.pause(0.001)
		
	apertIndex = np.random.permutation(nApert) # iterate at random order
	velocities *= momentum
	velocities -= finalObj

	if bMakeReset:
		if iter%resetEvery == 0 and iter <= resetTo: # Reset object, keep probes
			finalObj = GetRandomObj(ObjShape)
			velocities = cgpuarray(ObjShape)
			relative_iter = 1
			
	if bNormalizeEach:
		for j in np.int32(range(NumModes)):
			ModeSquared(aperture, fcachevector)
			jNorm = np.sqrt(1.0/np.sum(fcachevector.get()))
			ModeMultiply(aperture, j, np.float32(jNorm))
	elif bNormalizeAll:
		NormSquared(aperture, fcachevector)
		aperture *= np.float32(np.sqrt(1.0/np.sum(fcachevector.get())))
		
	for cont in range(nApert):
		aper =  apertIndex[cont]

		offsety = np.int32(np.round(roiList[aper][0]))
		offsetx = np.int32(np.round(roiList[aper][2]))  # XY ROI position
		roisizex = np.int32(np.round(roiList[aper][3]-roiList[aper][2])) # ROI span:  [ofx,ofx+sizex]
		roisizey = np.int32(np.round(roiList[aper][1]-roiList[aper][0])) # ROI span:  [ofy,ofy+sizey]
		objsizex = np.int32(np.round(ObjShape[1]))

		# Position Correction, choose random aperture as center
		if iPosCorrection > 0 and iter>100 and iter <= 2000 and iter%(50*iPosCorrection) == 0 and aper != 12:
			mini,minj = RunCorrection(iPosCorrection,roiList[aper],DifPads[aper],rspace,kspace,exitWave,buffer_exitWave,finalObj,offsetx,offsety,objsizex,roisizex,CopyFromROI,ExitwaveAndBuffer,ApplyDifPad,cufftplan,aperture,fcachevector)
			
			# Display pos corrections
			# if mini != 0 or minj != 0:
			#	print aper,[mini,minj]
			
			if roiList[aper][0] + minj >= 0 and roiList[aper][1] + minj < ObjShape[0]:
				roiList[aper][0] += minj
				roiList[aper][1] += minj
				offsety += np.int32(minj)
			if roiList[aper][2] + mini >= 0 and roiList[aper][3] + mini < ObjShape[1]:
				roiList[aper][2] += mini
				roiList[aper][3] += mini
				offsetx += np.int32(mini)

		CopyFromROI(rspace, finalObj, offsety, offsetx, roisizex, objsizex) # Copy roi to rspace
				
		if iter < 10 or relative_iter < 3 or (iter-1)%PerfLevel == 0: # Compute |Pmax|^2 and |Omax|^2
			ApertureAbs2(aperture,fcachevector)
			apertMax = fcachevector.get().max() * 0.2 + apertMax * 0.8
			
			ObjectAbs2(rspace,fcachevector)
			ObjMax[aper] = fcachevector.get().max() * 0.2 + ObjMax[aper]*0.8

		if bPhaseShift: # if bPhaseShift, shift rspace to desired (subpixel) position
			inplaceFractShift(rspace,-roiList[aper][2]+offsetx,-roiList[aper][0]+offsety,PhaseShiftFunc)
			
		ExitwaveAndBuffer(exitWave, buffer_exitWave, aperture, rspace) # Compute exitwaves
		cu_fft.fft(exitWave,kspace,cufftplan) # kspace = wave at detector
		ApplyDifPad(kspace,DifPads[aper],fcachevector) # replace amplitudes.
		cu_fft.ifft(kspace,exitWave,cufftplan,True)	# new exitwave

		Pmax2 = np.float32(apertMax)
		Omax2 = np.float32(ObjMax[aper])
		betaObj = np.float32(betaObj)
		betaAP = np.float32(betaApert)
		if iter<10:
			betaAP = np.float32(0)

		# Update exitwave/Probe in rspace and exitwave/rspace in probe
		UpdateProbeAndRspace(rspace, exitWave, buffer_exitWave, aperture, betaObj, Pmax2, betaAP, Omax2, np.float32(0.25))
		
		if bPhaseShift:		
			inplaceFractShift(rspace,roiList[aper][2]-offsetx,roiList[aper][0]-offsety,PhaseShiftFunc)
			
		# Update Obj with the new rspace
		ReplaceInObject(finalObj, rspace, offsety, offsetx, roisizex, objsizex)
			
		if (iter-1)%PerfLevel == 0: # Compute error. Cachevector has error calculated on ApplyDifPad
			lastError[aper] = fcachevector.get().sum()/difpadsums[aper]
		errorF[int(iter-1),aper] = lastError[aper]
			
	# Crop object where the probe doesnt touch
	if bCropObject and iter > CropSav and iter < iterations-CropSav:
		objsize = np.int32(ObjShape[0]*ObjShape[1])
		probethresh = np.float32(crop_threshold/roisizex/roisizey)
		CropObject(aperture, finalObj, roioffsetsgpu, nApert, objsizex, roisizex, probethresh, objsize, np.float32(crop_factor))
	
	velocities += finalObj
	finalObj += betaObj*velocities
	
	relative_iter = relative_iter + 1
	iter = iter + 1
	
	printw('Iter: ' + str(iter) + '. Error: ' + common.ToPercent(np.sum(lastError)/nApert) + '  ')
	
	# Impose orthogonalization of the probes
	if bRelaxedOrtho:
		ModeSquared(aperture, fcachevector, 0)
		JNorm = np.sum(fcachevector.get())
		for i in np.int32(range(1,NumModes)):
			ScalarProd(aperture, ccachevector, 0, i)
			alpha = np.sum(ccachevector.get())/jNorm
			ModeSub21(aperture, 0, i, np.complex64(alpha))
	elif bOrthogonal:
		for j in np.int32(range(NumModes-1)):
			ModeSquared(aperture, fcachevector, j)
			JNorm = np.sum(fcachevector.get())
			for i in np.int32(range(j+1,NumModes)):
				ScalarProd(aperture, ccachevector, j, i)
				alpha = np.sum(ccachevector.get())/jNorm
				ModeSub21(aperture, j, i, np.complex64(alpha))
		
plt.ioff()
print('')

curtime = time()-curtime
print "Done in " + str(int(curtime)) + "." + str(int(curtime*10)%10) + "s"
	
plt.subplot(2, 2, 1)
plt.title('Test')
imgplot = plt.imshow(finalObj.__abs__().get())

plt.subplot(2, 2, 2)
imgplot = plt.imshow(np.angle(finalObj.get()))		# Use phase and Hue
#imgplot = plt.imshow(common.CMakeRGB(finalObj.get())) 	# This is the full HSV version

plt.subplot(2, 2, 3) # Only displaying probes amplitude
plt.imshow( np.reshape(aperture.__abs__().astype(np.float32).get(), (NumModes*roisizey,roisizex)) )

plt.subplot(2, 2, 4)
plt.plot(np.log(errorF+1E-10)/np.log(10))
plt.ylabel('Fourier R-factor')
plt.xlabel('Iteration')
#plt.title('Worst R-factor: %s' %(errorF*100))
plt.tight_layout()
plt.show()

