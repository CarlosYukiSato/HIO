#	@author: giovanni.baraldi

from pymba import *
import time
import cv2
import numpy as np
from copy import deepcopy
from sys import exit
from os import listdir

def WriteImg(img, filename):
	#rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
        cv2.imwrite(filename, img)

Curfiles = []
Filename = ''
FOLDER = ''
exptimes = []

def UpdateFiles():
	global Curfiles
	global FOLDER
	global Filename
	global exptimes
	
	Curfiles = []
	for file in listdir(FOLDER):
       		Curfiles.append(file)

def IsFileInFolder(expt):
	global Curfiles
	global FOLDER
	global Filename
	global exptimes
	
	for file in Curfiles:
		if file[0:len(Filename+str(expt))] == Filename+str(expt):
			return True
	return False

def IsCaptureSequencyReady():
	global Curfiles
	global FOLDER
	global Filename
	global exptimes
	
	for expt in exptimes:
		if IsFileInFolder(expt) == False:
			return False
	print('CaptureSequency: ' + Filename + ' ready')
	return True

def GetFrameArray(ExpTimeArray, headername, imgformat):
	global Curfiles
	global FOLDER
	global Filename
	global exptimes
	
	FrameArray = []

	vimba = Vimba()
	vimba.startup()
	system = vimba.getSystem()
	if system.GeVTLIsPresent:
		system.runFeatureCommand("GeVDiscoveryAllOnce")
		time.sleep(0.2)
	camera = vimba.getCamera(vimba.getCameraIds()[0])
	camera.openCamera()
	camera.ExposureAuto = 'Off'
	camera.AcquisitionMode = 'SingleFrame'
	camera.AcquisitionFrameRateMode = 'Basic'
	camera.PixelFormat = 'Mono8'
	camera.AcquisitionFrameRate = 10
	camera.ExposureMode = 'Timed'
	frame = camera.getFrame()
	frame.announceFrame()
	camera.startCapture()
	
	lastexp = -1.0
	try:
		for expTime in ExpTimeArray:
			if expTime <= 0 or IsFileInFolder(expTime):
				continue

			print('Exposing ' + Filename + str(expTime))

			if expTime >= 2E5:
				if lastexp < 2E5:
					camera.AcquisitionFrameRate = 1
			elif expTime >= 5E4:
				if lastexp < 5E4:
					camera.AcquisitionFrameRate = 4

			camera.ExposureTimeAbs = expTime # Esta diferente do manual
			try:
				frame.queueFrameCapture()
			except: 
				print('Capture failed for: ' + str(expTime))
				quit()

        		camera.runFeatureCommand('AcquisitionStart')
			time.sleep(expTime/1.0E6)
       			camera.runFeatureCommand('AcquisitionStop')
			time.sleep(expTime/1.0E6)
			frame.waitFrameCapture()

			bytebuffer = frame.getBufferByteData()
			bytedata = np.ndarray( buffer=bytebuffer, dtype=np.uint8, 
						shape=(frame.height, frame.width) )

			WriteImg(bytedata, headername + str(expTime) + imgformat)
			lastexp = expTime

	except:
		print("Capture Failed")
	finally:
		camera.endCapture() 
		camera.revokeAllFrames()
		vimba.shutdown()

MAX_ESPOSURE = 1000000
FOLDER = 'Capilar/'

power = 0
expt = 0

exptimes = [31,62,93]

while expt < MAX_ESPOSURE:
	expt = int(250.0*pow(2.7,power))
	exptimes.append(expt)
	power += 0.10

def CaptureExposureTimes():
	bAllCaptured = False
	while bAllCaptured == False:
		UpdateFiles()
		GetFrameArray(exptimes, FOLDER+Filename, '.png')
		bAllCaptured = IsCaptureSequencyReady()

for Run in range(0,5):
	Filename = 'Dark' + str(Run) + '_'
	CaptureExposureTimes()
