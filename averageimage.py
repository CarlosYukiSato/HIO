#	@author: giovanni.baraldi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from copy import copy
from common import *

def GetAverageImage(images,errors): # Returns the image
	N = len(images)
	assert(len(errors) == N)

	bestrecs = [] # Stores best reconstructions
	besterrors = [] # Stores best errors

	MasterImg = images[np.argmin(errors)] # Use best reconstruction as reference
	finalimage = np.zeros(MasterImg.shape,dtype=np.complex64) # Stores final result
	weights = 0.0

	temperrors = np.asarray(errors,dtype=np.float32) + 0.0
	for k in range(0,int(np.sqrt(N))):
		best = np.argmin(temperrors)
		temperrors[best] = 1E15
		bestrecs.append(images[best])
		besterrors.append(errors[best])
		print('Chosen: ' + str(best) + ' with error: ' + str(errors[best]))	# Choose the best sqrt(2N), this number looks fine

	for k in range(0,len(bestrecs)):

		img1 = bestrecs[k]
		img2 = np.flip(img1,0)
		img2 = np.flip(img2,1) # Image may be flipped, so test it

		argy1,argx1,value1 = ArgCorrelationOf(MasterImg,img1)
		argy2,argx2,value2 = ArgCorrelationOf(MasterImg,img2)

		if value1 >= value2:
			print('Unflip: ' + str(argy1) + ' ' + str(argx1) + ' ' + str(value1))
			img1 = np.roll(img1,argy1,0)
			img1 = np.roll(img1,argx1,1)
			finalimage += img1*value1
			weights += value1	# Use correlation as weight, why not
		else:
			print('Flip: ' + str(argy2) + ' ' + str(argx2) + ' ' + str(value2))
			img2 = np.roll(img2,argy2,0)
			img2 = np.roll(img2,argx2,1)
			finalimage += img2*value2
			weights += value2

	finalimage /= weights

	return finalimage,MasterImg

################## EXAMPLE ##################

name = 'CentroEstrela5deg'
header = 'OSS_Files/' + name + '/Run'

ampname = '_rspace_amp_oss.npy'
phasename = '_rspace_phase_oss.npy'

N = 13
center = 480
size = 50

def Load(k):
	Amps = np.load(header + str(k) + ampname)
	Phases = np.load(header + str(k) + phasename)
	Img = Amps*np.exp(1j*Phases)
	return Img

images = [] # Stores all reconstructions
errors = [] # Stores all errors

IM_HALFSIZE = 480
DifPad = np.load('HDR/' + name + '.npy')[498-IM_HALFSIZE:498+IM_HALFSIZE,683-IM_HALFSIZE:683+IM_HALFSIZE]

for k in range(0,N):
	Img = Load(k)
	
	if True:
		Mask = np.zeros(Img.shape,dtype=np.bool)
		Mask[:,:] = False
		for j in range(-50,50):
			for i in range(-50,50):
				if(j*j + i*i < 2500):
					Mask[480 + j, 480 + i] = True

		error = Error_real(Img,Mask)
	else:
		error = Error_recip(Img,DifPad)

	errors.append(error)
	images.append(Img[center-size:center+size,center-size:center+size]) # We'll use a smaller version of each reconstruction

finalimage,bestimage = GetAverageImage(images,errors)

plt.subplot(2,2,1)
plt.title('Avg Amplitude')
plt.imshow(np.absolute(finalimage),cmap='jet')

plt.subplot(2,2,2)
plt.title('Avg Phases')
plot = plt.imshow(CMakeRGB(finalimage))	

plt.subplot(2,2,3)
plt.title('Best Amplitude')
plt.imshow(np.absolute(bestimage),cmap='jet')

plt.subplot(2,2,4)
plt.title('Best Phase')
plot = plt.imshow(CMakeRGB(bestimage))	

plt.show()
