import numpy as np
from PIL import Image
from numpy import absolute as abs

gdifpad = None
gsqdifpad = None
gdifsum = None
initialGuess = 10

def Noise(difpad,errorRate):
	return np.random.poisson(difpad/errorRate, difpad.shape)*errorRate

def RFactor(errorRate):
	global gdifpad
	global gsqdifpad
	global gdifsum
	
	difpad2 = Noise(gdifpad,errorRate)
	return np.sum(abs(gsqdifpad-np.sqrt(difpad2)))/gdifsum

def GetErrorRateNoise(rFact):
	global initialGuess
	
	errorR = initialGuess
	while RFactor(errorR) < rFact*0.01:
		errorR *= 16.0
	while RFactor(errorR) > rFact*0.01:
		errorR *= 1.0/16.0
	
	maxR = errorR * 16.0
	minR = errorR
	errorR = (minR+maxR)*0.5

	while abs(RFactor(errorR) - rFact*0.01) > rFact*1E-4:
		for k in range(0,3):
			if RFactor(errorR) > rFact*0.01:
				maxR = errorR
			else:
				minR = errorR
			errorR = (minR+maxR)*0.5
			
	initialGuess = errorR
	return errorR
		
def GetNoise(difpad,rFact):
	global gdifpad
	global gsqdifpad
	global gdifsum
	
	gdifpad = difpad+0
	gsqdifpad = np.sqrt(difpad)
	gdifsum = np.sum(gsqdifpad)
	errorGuess = GetErrorRateNoise(rFact)
	return Noise(difpad,errorGuess)

def GetNumPhotons(difpad,nphotons):
	errorRate = np.sum(difpad)/nphotons
	return Noise(difpad,errorRate)

