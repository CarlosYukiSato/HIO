import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import common
from numpy.fft import fftshift as fftshift
import noise
from common import printw
from common import Format

def MakePosErrors(roiList,errormaxsize):
	newroilist = []
	errors = []
	errm = errormaxsize + 0.999
	for ROI in roiList:
		errx = int(2*errm*np.random.rand()-errm)
		erry = int(2*errm*np.random.rand()-errm)
		newroi = [ROI[0]+erry,ROI[1]+erry,ROI[2]+errx,ROI[3]+errx]
		newroilist.append(newroi)
		errors.append([errx,erry])
	return newroilist,errors
	

def makeCircAperture(Diameter,imagSize): #Creates a flat circular aperture
	Raio = Diameter/2
	x = np.arange(-(imagSize[1]-1.0)/2 , (imagSize[1]+1.0)/2, 1.0)
	y = np.arange(-(imagSize[0]-1.0)/2 , (imagSize[0]+1.0)/2, 1.0)
	xx, yy = np.meshgrid(x, y, sparse=True)
	aperture = np.zeros(imagSize,dtype=np.complex64)
	aperture[xx**2 + yy**2 < Raio**2] = 1 #Circle equation        
	    
	return aperture
	
def makeSincAperture(Diameter,imagSize):	
	x = np.arange(-imagSize[1]//2, imagSize[1]//2, 1.0) + 0.5
	y = np.arange(-imagSize[0]//2, imagSize[0]//2, 1.0) + 0.5
	xx, yy = np.meshgrid(x, y, sparse=True)
	
	rr = np.sqrt(xx**2+yy**2)/Diameter
	aperture = np.sin(np.pi*rr)/rr
	aperture[rr > 1] = 0
	
	return aperture.astype(np.complex64)

def makeGaussAperture(sigmaValue,imagSize,Threshold=0.5): #Creates Gaussian probe
	x = np.arange(-(imagSize[1]-1)/2.0 , (imagSize[1]+1)/2.0, 1)
	y = np.arange(-(imagSize[0]-1)/2.0 , (imagSize[0]+1)/2.0, 1)

	xx, yy = np.meshgrid(x, y, sparse=True)
	aperture = np.exp(	-0.5*( (yy)**2+(xx)**2 )/sigmaValue**2	)#gaussian
	aperture = aperture / np.max(aperture)
	aperture[aperture<Threshold] = 0

	return aperture.astype(np.complex64)

def MakeGaussApertures(sigmas,imagApert,thresh,amps=[1,0.1,0.1,0.1,0.1],rolls=[]):
	aperture = []
	for sigma in range(len(sigmas)):
		AP = makeGaussAperture(sigmas[sigma],imagApert,thresh)
		AP /= np.sqrt(np.sum(np.absolute(AP)**2))
		if sigma < len(rolls):
			AP = np.roll(np.roll(AP,rolls[sigma][1],1),rolls[sigma][0],0)
		aperture.append( (amps[sigma]*AP).astype(np.complex64) )
	return aperture

def MakeCircApertures(diameters,imagApert,amps=[1,0.1,0.1,0.1,0.1],rolls=[]):
	aperture = []
	for diam in range(len(diameters)):
		AP = makeCircAperture(diameters[diam],imagApert)
		#AP /= np.sqrt(np.sum(np.absolute(AP)**2))
		if diam < len(rolls):
			AP = np.roll(np.roll(AP,rolls[diam][1],1),rolls[diam][0],0)
		aperture.append( (amps[diam]*AP).astype(np.complex64) )
	return aperture


def MakeSincApertures(diameters,imagApert,amps=[1,0.1,0.1,0.1,0.1],rolls=[]):
	aperture = []
	for diam in range(len(diameters)):
		AP = makeSincAperture(diameters[diam],imagApert)
		AP /= np.sqrt(np.sum(np.absolute(AP)**2))
		if diam < len(rolls):
			AP = np.roll(np.roll(AP,rolls[diam][1],1),rolls[diam][0],0)
		aperture.append( (amps[diam]*AP).astype(np.complex64) )
	return aperture

def GetRois(numstepsx,numstepsy,stepsize,imagApert):
	roiList = []
	for posy in range(0,numstepsy):
		for posx in range(0,numstepsx):
			ROI = [posy*stepsize,posy*stepsize + imagApert[0],posx*stepsize,posx*stepsize + imagApert[1]]
			for kkk in range(0,4):
				ROI[kkk] = np.int32(ROI[kkk])
			roiList.append(ROI)
	return roiList
	
def ApplyJitter(roilist,size,offset):
	for j in range(len(roilist)):
		jiy = int(np.random.rand()*(2*size-0.0001)-size) + offset
		jix = int(np.random.rand()*(2*size-0.0001)-size) + offset
		roilist[j][0] += jiy
		roilist[j][1] += jiy
		roilist[j][2] += jix
		roilist[j][3] += jix

def NearestCut(image2,iBinning):
	newimage = image2+0
	image = image2+0
	newimage[:,:] = 0
	
	for m in range(0,image.shape[0]//iBinning):
		newimage[m,:] = image[iBinning*m+iBinning//2,:]
		
	image[:,:] = 0
	for n in range(0,image.shape[1]//iBinning):
		image[:,n] = newimage[:,iBinning*n+iBinning//2]
				
	return image[0:image.shape[0]//iBinning,0:image.shape[1]//iBinning]
	
def Bin(image2,iBinning):
	newimage = image2+0
	image = image2+0
	
	newimage[:,:] = 0
	for m in range(iBinning):
		newimage += np.roll(image,m-iBinning//2,1)
		
	image[:,:] = 0
	for m in range(iBinning):
		image += np.roll(newimage,m-iBinning//2,0)
		
	return NearestCut(image*1.0/iBinning**2,iBinning)

def GetDifPads(model, probe, rois, ErrorPercent=0): # 0.0000167 for Sinc of size 5
	DifPads = []
	difpadsums = []
	nPhotons = 0
	
	print 'Generating Difpads'
	for ROI in rois:
		diflist = np.absolute(np.fft.fft2(probe*model[ROI[0]:ROI[1],ROI[2]:ROI[3]]))**2
				
		if ErrorPercent > 0:
			diflist = noise.GetNoise(diflist,ErrorPercent)
			nPhotons += np.sum(diflist)/noise.initialGuess
					
		DifPads.append( diflist.astype(np.float32) )
		#difpadsums.append(np.sum(diflist))
	
	if ErrorPercent > 0:
		print '\nnPhotons:',Format(1.0*nPhotons/probe.shape[0]/probe.shape[1]/len(rois))
	
	return DifPads,nPhotons

def LoadLenaModel():
	im = Image.open('Medidas/Lena.tiff')
	Lena = np.array(im)
	Lena = 0.2989 * Lena[:,:,0] + 0.5870 * Lena[:,:,1] + 0.1140 * Lena[:,:,2]
	Lena /= Lena.max()

	im = Image.open('Medidas/boatlake.tiff')
	BoatLake = np.array(im)
	BoatLake = 0.2989 * BoatLake[:,:,0] + 0.5870 * BoatLake[:,:,1] + 0.1140 * BoatLake[:,:,2]
	BoatLake = BoatLake - np.min(BoatLake)
	BoatLake *= BoatLake
	BoatLake = 1.0 * BoatLake / np.max(BoatLake) - 0.5
	del(im)

	return (Lena * np.exp(1j * BoatLake * np.pi)).astype(np.complex64)

def LoadExtendedLenaModel():
	finalModel2 = LoadLenaModel() # Get default model. Replace this by your image.
	finalModel = np.zeros((finalModel2.shape[0]*2,finalModel2.shape[1]*2),dtype=np.complex64)
	finalModel[0:finalModel2.shape[0],0:finalModel2.shape[1]] = finalModel2[:,:]
	finalModel[finalModel2.shape[0]:,0:finalModel2.shape[1]] = finalModel2[:,:]**2/finalModel2.max()
	finalModel[0:finalModel2.shape[0],finalModel2.shape[1]:] = finalModel2[:,:]**2/finalModel2.max() + finalModel2[:,:]
	finalModel[finalModel2.shape[0]:,finalModel2.shape[1]:] = finalModel2[:,:]
	return finalModel

def LoadNISTModel():
	fireimage = np.asarray(Image.open('Medidas/FireNIST.jpg'))[200:1224,:,:]
	fireimage = fireimage[:,:,0] + fireimage[:,:,1] + fireimage[:,:,2]
	fireimage -= fireimage.min()
	fireimage = 1.0*fireimage/fireimage.max()
	flipped = np.flip(fireimage,0)
	realimage = np.zeros((2048,2048),dtype=np.float32)
	realimage[0:1024,0:2000] = fireimage[:,:]
	realimage[1024:2048,0:2000] = flipped[:,:]
	
	kgimage = np.asarray(Image.open('Medidas/kgNIST.png'))[1000:3048,1600:3648,:]
	kgimage = kgimage[:,:,0] + kgimage[:,:,1] + kgimage[:,:,2]
	kgimage -= kgimage.min()
	kgimage = 2.0*kgimage/kgimage.max() - 1.0
	
	image = realimage * np.exp(2j*np.pi*kgimage)
	return image.astype(np.complex64)

def MakeObjShapeFromROIS(rois):
	topx = 0
	topy = 0
	for roi in rois:
		topx = max(topx,roi[3])
		topy = max(topy,roi[1])
	ObjShape = [topy,topx]
	for m in range(0,2): # Make the shape a power of two times (2,3 or 5). Obs: FFTW is fine with larger primes.
		pow2 = 1
		while pow2 < ObjShape[m]:
			pow2 *= 2
		if 5*pow2//8 > ObjShape[m]:
			pow2 = 5*pow2//8
		elif 3*pow2//4 > ObjShape[m]:
			pow2 = 3*pow2//4
		ObjShape[m] = pow2
	ObjShape = (max(ObjShape[0],ObjShape[1]),max(ObjShape[0],ObjShape[1]))
	return ObjShape

def BinDifpad(difpad,iBinning): # For the experimental setup
	diftemp1 = difpad+0
	if np.sum(diftemp1<0) > 0:
		print 'Warning:', j,i
	diftemp1 = diftemp1**2
	diftemp2 = np.zeros(diftemp1.shape,dtype=np.float32)
				
	for m in range(iBinning):
		diftemp2 += np.roll(diftemp1,m,1)
	diftemp1 = diftemp2+0
	diftemp2 = np.zeros(diftemp1.shape,dtype=np.float32)
	for m in range(iBinning):
		diftemp2 += np.roll(diftemp1,m,0)
		
	for m in range(0,diftemp1.shape[0]//iBinning):
		diftemp1[m,:] = diftemp2[iBinning*m,:]	
	diftemp2 = np.zeros(diftemp1.shape,dtype=np.float32)	
	for n in range(0,diftemp1.shape[1]//iBinning):
		diftemp2[:,n] = diftemp1[:,iBinning*n]
				
	diftemp2 = diftemp2[0:diftemp1.shape[0]//iBinning,0:diftemp1.shape[1]//iBinning]
	return diftemp2


	
