#	@author: giovanni.baraldi

import numpy as np
from matplotlib.colors import hsv_to_rgb
import sys
from decimal import Decimal

def Format(numberin,ndec=2):
	return ("{:."+str(ndec)+"E}").format(numberin)

def ToPercent(value):
	if value > 1E-2:
		return "{:2.2f}".format(100.0*value) + ' %'
	else:
		return "{:.2E}".format(value)

def printw(stri):
	sys.stdout.write('\r' + str(stri))
	sys.stdout.flush()

def MakeRGB(Amps,Phases,bias=0): # Make RGB image from amplitude and phase
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
	return Amps,Phases

def CMakeRGB(ComplexImg,bias=0.15):
	Amps,Phases = SplitComplex(ComplexImg)
	return MakeRGB(Amps,Phases,bias)

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
	
def getline(diff,d,size):
	Border = diff[size-d,:] + diff[size+d,:] + diff[:,size+d] + diff[:,size-d]
	return Border[size-d:size+d+1]
	
def SemiFRC(img1ol,img2ol):
	size = img1ol.shape[0]//2
	
	img1 = np.fft.fftshift(np.fft.fft2(img1ol))
	img2 = np.fft.fftshift(np.fft.fft2(img2ol))

	mmc = img1*np.conj(img2)
	sub = (abs(img1) - abs(img2))**2
	
	img1 = abs(img1)**2
	img2 = abs(img2)**2
	
	FRC = []
	SNR = []
	
	for d in range(size):
		FRC.append( np.sum(getline(mmc,d,size))/np.sqrt( np.sum(getline(img1,d,size)) * np.sum(getline(img2,d,size)) ) )
		SNR.append( np.sum(getline(img1,d,size))/(np.sum(getline(sub,d,size))+1E-20) )
		
	SNR = np.log(abs(np.asarray(SNR)))/np.log(10)
	FRC = abs(np.asarray(FRC))
	
	SNR[SNR<0] = 0
	SNR[SNR>1] = 1
	
	return FRC,SNR
		
		
	
	
	
	

