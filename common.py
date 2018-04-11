#	@author: giovanni.baraldi

import numpy as np
from matplotlib.colors import hsv_to_rgb
import sys

def printw(stri):
	sys.stdout.write('\r' + stri)
	sys.stdout.flush()

def MakeRGB(Amps,Phases,bias): # Make RGB image from amplitude and phase
	HSV = np.zeros((Amps.shape[0],Amps.shape[1],3),dtype=np.float32)
	normalizer = (1.0-bias)/Amps.max()
	HSV[:,:,0] = Phases[:,:]
	HSV[:,:,1] = 1
	HSV[:,:,2] = Amps[:,:]*normalizer + bias
	return hsv_to_rgb(HSV)

def SplitComplex(ComplexImg):
	Phases = np.angle(ComplexImg)		# Phases in range [-pi,pi]
	Phases = Phases*0.5/np.pi + 0.5
	Amps = np.absolute(ComplexImg)
	Amps /= np.max(Amps) # No particular reason
	return Amps,Phases

def CMakeRGB(ComplexImg):
	Amps,Phases = SplitComplex(ComplexImg)
	return MakeRGB(Amps,Phases,0.1)

def Error_real(rspace,Support):
	rReal = rspace.real**2
	sumreal = np.sum(rReal)
	rReal[Support==True] = 0
	return np.sqrt(sumreal/np.sum(rReal))

def Error_recip(kspace,DifPad):
	fixed = FixToPad(DifPad,kspace)
	err = fixed-DifPad
	return np.sqrt(np.sum(err*err.conj()).real)


def ArgCorrelationOf(fixedimg,newimg): #Returns x,y shift and max correlation found
	fft1 = np.fft.fft2(fixedimg.astype(np.complex64))
	fft2 = np.fft.fft2(newimg.astype(np.complex64))

	Kernel = np.zeros(fft1.shape,dtype=np.complex64) # To smooth out the correlation
	cx = Kernel.shape[1]//2
	cy = Kernel.shape[0]//2

	for j in range(-10,+10):
		for i in range(-10,+10):
				Kernel[cy+j,cx+i] = np.exp( -0.25*(j*j+i*i) )
	Kernel = np.fft.fft2(np.fft.fftshift(Kernel))

	R = fft1*fft2.conj()
	Rmod = np.absolute(R)
	R = R/Rmod
	R[Rmod==0] = 0
	R *= Kernel # Apply gaussian blur to avoid artifacts
	R = np.absolute(np.fft.ifft2(R)) # R = F-1{ F(r1)F*(r2) / | F(r1)F*(r2) | }

	arg = np.argmax(R)
	dy = arg//R.shape[0]	 # np.roll(x,arg) works, only using (dy,dx) for debugging
	dx = arg - dy*R.shape[0]

	return dy,dx,np.max(R)

def FixToPad(DifPad,img): # returns fft with the same center and overall sum as DifPad
	dx,dy,corr = ArgCorrelationOf(img,DifPad)
	img = np.roll(img,dy,0)
	img = np.roll(img,dx,1) # Center fft to difpad

	sumdif = np.sum(img)
	img /= sumdif # Normalize
	return img

