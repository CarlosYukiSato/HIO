import numpy as np
from PIL import Image
import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath
import matplotlib.pyplot as plt
import common
from numpy.fft import fftshift as fftshift
import skcuda.fft as cu_fft
import noise
from common import printw

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
		AP /= np.sqrt(np.sum(np.absolute(AP)**2))
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

def GetRandomObj(ObjShape):
	qsizex = ObjShape[0]//4
	qsizey = ObjShape[1]//4
	
	#randomobj = 1E-2*(np.random.rand(2*qsizey,2*qsizex) + 1j * np.random.rand(2*qsizey,2*qsizex))
	#zeroobj = np.zeros(ObjShape,dtype=np.complex64)
	#zeroobj[qsizey:3*qsizey, qsizex:3*qsizex] = randomobj[:,:]
	
	#return gpuarray.to_gpu(zeroobj)
	return gpuarray.to_gpu((np.random.rand(ObjShape[0],ObjShape[1])+1j*np.random.rand(ObjShape[0],ObjShape[1])).astype(np.complex64))

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

def GetDifPads(model, probe, rois, ErrorPercent=20, superres=1, iBinning=1): # 0.0000167 for Sinc of size 5
	DifPads = []
	difpadsums = []
	nPhotons = 0
	counterR = 0
	
	for lROI in rois:
		printw(str(counterR) + ' of ' + str(len(rois)))
		counterR += 1
		ROI = [int(superres*lROI[0]),int(superres*lROI[1]),int(superres*lROI[2]),int(superres*lROI[3])]
		diflist = np.zeros(probe[0].shape,dtype=np.float32)
		for mode in range(len(probe)):
			diflist += np.absolute(np.fft.fft2(probe[mode]*model[ROI[0]:ROI[1],ROI[2]:ROI[3]]))**2
		
		if iBinning!=1:
			diflist = Bin(diflist,iBinning)
		diflist = fftshift(diflist)
		
		hpx = diflist.shape[1]//2
		hpy = diflist.shape[0]//2
		lpx = int((lROI[3]-lROI[2])/2)
		lpy = int((lROI[1]-lROI[0])/2)
		diflist = diflist[hpy-lpy:hpy+lpy, hpx-lpx:hpx+lpx]
		
		diflist = fftshift(diflist)
		
		if ErrorPercent > 0:	
			diflist = noise.GetNoise(diflist,ErrorPercent)
			nPhotons += np.sum(diflist)/noise.initialGuess
		diflist = np.sqrt(diflist)
		
		DifPads.append(gpuarray.to_gpu(diflist.astype(np.float32))) ## WHY???
		difpadsums.append(np.sum(diflist))
	
	if ErrorPercent > 0:
		expo = int(np.log(nPhotons)/np.log(10))
		nPhotons = int(nPhotons)//10**expo
		print 'NPhotons:',nPhotons,'E'+str(expo)
	
	return DifPads,difpadsums

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
	BoatLake = 2.0 * BoatLake / np.max(BoatLake) - 1.0
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

def cgpuarray(shape):
	return gpuarray.zeros(shape,dtype=np.complex64)
def fgpuarray(shape):
	return gpuarray.zeros(shape,dtype=np.float32)
def carray(shape):
	return np.zeros(shape,dtype=np.complex64)
def farray(shape):
	return np.zeros(shape,dtype=np.float32)

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

kxx = None
kyy = None
plan = None
FT = None
ftshape = None

def Cache(shape):
	global kxx
	global kyy
	global plan
	global FT
	global ftshape
	
	if ftshape != shape:
		ftshape = (shape[0],shape[1])
		ddrange = 2.0*np.pi*np.arange(-shape[0]//2,shape[0]//2,1)
		kxx,kyy = np.meshgrid(ddrange,ddrange)
		kxx = gpuarray.to_gpu(np.fft.fftshift(kxx/shape[1]).astype(np.float32))
		kyy = gpuarray.to_gpu(np.fft.fftshift(kyy/shape[0]).astype(np.float32))
		
		plan = cu_fft.Plan(shape, np.complex64, np.complex64, batch=1, idist=shape[0]*shape[1], odist=shape[0]*shape[1])
		FT = cgpuarray(shape)

def PhaseShift(img,dx,dy):
	ddrange = 2.0*np.pi*np.arange(-img.shape[0]//2,img.shape[0]//2,1)
	kxx,kyy = np.meshgrid(ddrange,ddrange)
	kxx = np.fft.fftshift(kxx/img.shape[1])
	kyy = np.fft.fftshift(kyy/img.shape[0])
	
	FT = np.fft.fft2(img)
	FT *= np.exp(1j*(kxx*dx + kyy*dy))
	return np.fft.ifft2(FT)

def inplaceFractShift(img,dx,dy,PhaseShiftFunc,bInverse=False):
	if dx==0 and dy==0:
		return
	global plan
	global FT
		
	Cache(img.shape)
	cu_fft.fft(img,FT,plan)
	PhaseShiftFunc(FT,kxx,kyy,np.float32(dx),np.float32(dy))
	cu_fft.ifft(FT,img,plan,True)

def FractShift(src,dest,dx,dy,PhaseShiftFunc):
	if dx==0 and dy==0:
		return
	global plan
	global FT
		
	Cache(src.shape)
		
	cu_fft.fft(src,FT,plan,PhaseShiftFunc)
	PhaseShift(FT,dx,dy)
	cu_fft.ifft(FT,dest,plan,True)

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

def CenterDifpad(difpad,IM_HALFSIZE): # For the experimental setup
	IndexMax = np.argmax(difpad)
	IM_CENTERY = IndexMax//difpad.shape[1]
	IM_CENTERX = IndexMax - IM_CENTERY*difpad.shape[1]
	Dif = difpad[IM_CENTERY-IM_HALFSIZE:IM_CENTERY+IM_HALFSIZE,IM_CENTERX-IM_HALFSIZE:IM_CENTERX+IM_HALFSIZE]
	return np.sqrt(np.fft.ifftshift(Dif))

# For the experimental setup
def GetRoisAndDifPads(globalname,numstepsx,numstepsy,stepsizex,stepsizey,iBinning,IM_HALFSIZE,offsetx=0,offsety=0,bInverted=False): 
	DifPads = []
	difpadsums = []
	roiList = []
	for j in range(0,numstepsy):
		for i in range(0,numstepsx):
			difpad = np.load(globalname + str(j) + '_X' + str(i) + '.npy')
			processedDif = CenterDifpad(BinDifpad(difpad,iBinning),IM_HALFSIZE)
			DifPads.append(gpuarray.to_gpu(processedDif.astype(np.float32)))
			difpadsums.append(np.sum(processedDif))
		
			ofy = stepsizey*j + offsety
			ofx = stepsizex*i + offsetx
			if bInverted:
				ofy = stepsizey*(numstepsy-j-1) + offsety
		
			roiList.append([ofy,ofy+2*IM_HALFSIZE,ofx,ofx+2*IM_HALFSIZE])
			
	return DifPads,difpadsums,roiList
	
	
