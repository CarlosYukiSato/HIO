'''
Implementation of the Maximum Likelihood algorithm by P. Thibault and M. Guizar-Sicairos (2012).
Meant for refinement of ptycographic phase retriaval.
author: Giovanni L. Baraldi

TODO: GPU Implementation
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import common

from numpy import roll
from numpy import absolute as abs
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from time import time
from common import printw
from common import Format as ToScience
from MaxLikelyHelper import *

def getcoord(roi):	# Obj to probe coordinates
	ofy = np.int32(np.round(roi[0]))
	ofx = np.int32(np.round(roi[2]))
	size = np.int32(np.round(roi[3]-roi[2]))
	return ofy,ofx,size

def CutObj(Object,roi): # Get section of obj
	ofy,ofx,size = getcoord(roi)
	return Object[ofy:ofy+size,ofx:ofx+size]
	
def Random(imgApert):
	return 2.0*np.random.rand(imagApert[0],imagApert[1])-1.0
	
def RandomPerturbation(Object, Radius, roiList, scale=5): # Add random errors to obj
	imagApert = (roiList[0][1]-roiList[0][0],roiList[0][3]-roiList[0][2])
	smallcircle = MakeCircApertures([Radius],imagApert)[0]
	for roi in roiList:
		ofy,ofx,size = getcoord(roi)
		Object[ofy:ofy+size,ofx:ofx+size] += scale*smallcircle*(Random(imagApert) + 1j*Random(imagApert))

np.random.seed(1)
iterations = 200
imagApert = np.int32(128) 
imagApert = (imagApert,imagApert)

#roiList = GetRois(11,11,23,imagApert) + GetRois(9,9,31,imagApert)
roiList = GetRois(9,9,32,imagApert)
ApplyJitter(roiList,5,16) 	# offset ROI and apply random jitter to avoid artifacts
ObjShape = (512,512)

#Model = LoadLenaModel()
Model = LoadNISTModel()
aperture = MakeCircApertures([64],imagApert)[0] + 1j*MakeCircApertures([32],imagApert)[0]
DifPads,nPhotons = GetDifPads(Model,aperture,roiList,30.0)

nApert = np.int32(len(DifPads))

Model = Model[:ObjShape[0],:ObjShape[1]]
Object = Model+0
	
# Add random errors to obj
RandomPerturbation(Object,32,roiList)
		
def GradChi(difpad,rspace,probe):
	phibar = fft2(rspace*probe)
	Int = abs(phibar)**2
	
	#Chibar = (Int-difpad)*phibar/(Int+1E-5) 		# Original version
	Chibar = 2*(1-np.sqrt(difpad/(Int+1E-5)))*phibar 	# Better
	
	return ifft2(Chibar)
	
def Gradients(difpads,Object,probe,roiList):
	gradobj = np.zeros(ObjShape,dtype=np.complex64)
	gradprobe = np.zeros(imagApert,dtype=np.complex64)
	
	for aper in range(nApert):	
		ofy,ofx,size = getcoord(roiList[aper])
		
		rspace = CutObj(Object,roiList[aper])
		chi = GradChi(difpads[aper],rspace,probe)
		
		# ! Paper says chi is the one to be conjugated but it doesnt work
		gradobj[ofy:ofy+size,ofx:ofx+size] += chi*np.conj(probe)
		gradprobe += chi*np.conj(rspace)
		
	return -gradobj/nApert,-gradprobe/nApert
	
def GetErr(exitW,difpad):
	return np.sum( abs( abs(fft2(exitW))**2-difpad ) )

def LineSearch(difpads,Object,probe,roiList,gradO, gradP, numsearches=3): # Remember to try this on ePIE
	
	objtrace = 1.0 # Acumulative step
	probetrace = 1.0 # Acumulative step
	
	for apIdx in range(numsearches):		# Pick <numsearches> random difpads for line search
		aper = int(np.random.rand()*nApert) 	# Each difpad will vote on the best step size
	
		exitW = CutObj(Object,roiList[aper])*probe
		gradW = CutObj(gradO,roiList[aper])*probe + CutObj(Object,roiList[aper])*gradP
	
		difpad = difpads[aper]
	
		stepct = 0.1 # Minimum step size
	
		minobjj = stepct
		minobjerr = 1E15

		for j in range(0,10):	 # exponential line search
			thiserr = GetErr( exitW + stepct * 2**j * gradW , difpad )
			if thiserr < minobjerr:
				minobjerr = thiserr
				minobjj = stepct * 2**j
		objtrace *= minobjj

	return np.power(objtrace,1.0/numsearches)	 # Geometric average looks more stable than arithmetic

iter = 1
useConjugate = False
curtime = time()

plt.ion()
plt.show()

aperture = abs(aperture).astype(np.complex64)

while (iter < iterations):
	#if iter%10 == 0 or (iter<10 and iter%4==0):
	if iter%100 == 0:
		plt.subplot(1,2,1)
		plt.imshow(abs(Object[32:256,32:256]))
		plt.subplot(1,2,2)
		plt.imshow(abs(aperture))
		plt.pause(0.001)
				
	#gradobj,gradprobe = Gradients(DifPads,Object,aperture,roiList)
	gradobj,gradp = Gradients(DifPads,Object,aperture,roiList)
	if iter < 10:
		gradp *= 0
		
	''' This one works best '''
	#'''
	gradocopy = gradobj+0
	gradpcopy = gradp+0
		
	if useConjugate == True: 	# beta = <g,g - g-1> / <g-1,g-1>
		
		# Paper says to rescale probe and object.
		# Best idea is to just ignore probe gradients when computing beta.
	
		s2 = np.sum(abs(gradobj)**2)/(np.sum(abs(gradp)**2)+1E-10)
	
		beta = np.sum( np.conj(gradobj)*(gradobj-prevgradO) )
		beta += s2 * np.sum( np.conj(gradp)*(gradp-prevgradP) )
		beta /= np.sum(abs(prevgradO)**2) + s2 * np.sum(abs(prevgradP)**2) + 1E-10
		
		gradobj += beta*prevUpdO
		gradp = gradp*np.sqrt(s2) + beta*prevUpdP 	
		#gradp = gradp*np.sqrt(s2) + beta*prevgradP
		
	prevgradO = gradocopy+0
	prevgradP = gradpcopy+0
	#'''
	
	''' # This one is faster '''
	'''
	if useConjugate == True:	# This usually works best
		beta = np.sum( np.conj(gradobj)*(gradobj-prevUpdO) ) / np.sum(abs(prevUpdO)**2)
		gradobj += beta*prevUpdO
		gradp = 0.1*gradp + beta*prevUpdP
	'''
	prevUpdO = gradobj+0
	prevUpdP = gradp+0
	
	searchval = LineSearch(DifPads,Object,aperture,roiList,gradobj,gradp)
	print 'Search:',searchval

	Object += searchval*gradobj
	aperture += searchval*gradp
	
	objabs = np.sqrt(np.sum(abs(Object[50:150,50:150])**2))
	ErrorR = np.sqrt( np.sum( (abs(Model[50:150,50:150])-abs(Object[50:150,50:150]))**2 ) )/objabs
	
	print 'Iter:',iter,'. Error:',ToScience(ErrorR)	# Realspace error
	iter = iter + 1
	useConjugate = True
	
	# Regularization -> Laplace
	'''
	Nm = nApert * imagApert[0]*imagApert[1]
	Npix = ObjShape[0]*ObjShape[1]		# Maybe use something similar to the CropObject mask
	
	uK = 1.25E-3 * Nm * nPhotons / Npix**2 	# u/K. Looks wierd to me but I believe the paper
	
	#Laplace
	Object += uK*( roll(Object,1,0) + roll(Object,-1,0) + roll(Object,1,1) + roll(Object,-1,1) - 4*Object )
	'''
		
plt.ioff()
print('')

FRC,SNR = common.SemiFRC(Object[50:278,50:278],Model[50:278,50:278])

plt.subplot(1,3,1)
plt.title('Poisson')
plt.plot( FRC )
plt.plot( SNR )
plt.subplot(2,3,2)
plt.imshow(abs(Object[50:278,50:278]))
plt.subplot(2,3,3)
plt.imshow(np.angle(Object[50:278,50:278]))
plt.subplot(2,3,5)
plt.imshow(abs(Model[50:278,50:278]))
plt.subplot(2,3,6)
plt.imshow(np.angle(Model[50:278,50:278]))
plt.show()

curtime = time()-curtime
print "Done in " + str(int(curtime)) + "." + str(int(curtime*10)%10) + "s"

