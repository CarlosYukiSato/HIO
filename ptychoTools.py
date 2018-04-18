# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:47:01 2018

@author: sato

Ptycho tools
"""

import numpy as np

def makeCircAperture(Diameter,imagSize,plot=False):
    """
    Creates a flat circular aperture
    """    
    Raio = Diameter/2
    x = np.arange(-(imagSize[1]-1.0)/2 , (imagSize[1]+1.0)/2, 1.0)
    y = np.arange(-(imagSize[0]-1.0)/2 , (imagSize[0]+1.0)/2, 1.0)
    xx, yy = np.meshgrid(x, y, sparse=True)
    aperture = np.zeros(imagSize,dtype=np.complex64)
    aperture[xx**2 + yy**2 < Raio**2] = 1 #Circle equation
    
    if plot:
        from matplotlib.pyplot import imshow
        imshow(aperture)
        
    
    return aperture
    
###############################################################################

def makeGaussAperture(sigmaValue,imagSize,plot=False):
    """
    Creates Gaussian probe, so far it is not truncated and you will have values up until the edge
    It would be better if we zero the probe after some value (3 sigma?) <= use makeCircAperture for mask    
    """
    
    x = np.arange(-(imagSize[1]-1.)/2 , (imagSize[1]+1.)/2, 1.)
    y = np.arange(-(imagSize[0]-1.)/2 , (imagSize[0]+1.)/2, 1.)

    x[x<0.5] = 0
    y[y<0.6] = 0
    xx, yy = np.meshgrid(x, y, sparse=True)
    aperture = np.exp(-(((np.sqrt((yy)**2+(xx)**2)**2))/(2*sigmaValue**2)))#gaussian 
    aperture = aperture / np.max(aperture)
 
    if plot:
        from matplotlib.pyplot import imshow
        imshow(aperture)   
    
    
    return aperture
    
###############################################################################    

def makePositions(rangeMin,rangeMax,numStep,stepSize,jiter=1):
    """
    Calculates the probe position for a given range
    positions = (x,y)
    """
    
    if isinstance(numStep, (int, np.integer)):
        numStep = [numStep,numStep]
        
    if isinstance(rangeMin, (int, np.integer)):
        rangeMin = [rangeMin,rangeMin]
        
    if isinstance(rangeMax, (int, np.integer)):
        rangeMax = [rangeMax,rangeMax]
    
    
    positionsY = np.linspace(rangeMin[0],rangeMax[0], num=numStep[0],dtype=float)
    positionsX = np.linspace(rangeMin[1],rangeMax[1], num=numStep[0],dtype=float)
 
    cont = 0
    positions = np.zeros([np.size(positionsY)*np.size(positionsX),2])    
    for conty in range(np.size(positionsX)):
        for contx in range(np.size(positionsY)):
            positions[cont , 0] = positionsX[contx]
            positions[cont , 1] = positionsY[conty]
            cont = cont +1 
            
    positions = positions * stepSize
    return positions
    
###############################################################################    
    
def calcDifPad(model,aperture,positions): 
    
    modelSize = np.array(model.shape)
    halfModelSize = np.ceil(modelSize/2)    
    
    apertSize = np.array(aperture.shape)
    halfApertSize = np.ceil(apertSize/2)
    
    DifPads = []

    
    for contP in range(np.size(positions,0)):
        realSpace = model
        realSpace = np.roll(realSpace,int(-positions[contP,1]),0)
        realSpace = np.roll(realSpace,int(-positions[contP,0]),1)
        realSpace = realSpace[int(halfModelSize[0]-halfApertSize[0]):int(halfModelSize[0]+halfApertSize[0]),:]
        realSpace = realSpace[:,int(halfModelSize[1]-halfApertSize[1]):int(halfModelSize[1]+halfApertSize[1])]
        DifPads.append( np.absolute(np.fft.fftshift(np.fft.fft2(realSpace * aperture))).astype(np.float32) )
        
        
    return DifPads
        
###############################################################################        

def createFinalObj(DifPads,positions,pixelsize=1,centerPositions=0):
    
    positions = np.round(positions / pixelsize)
    
    modelSize = np.array((len(DifPads),DifPads[0].shape[0],DifPads[0].shape[1]))
    eastSize = np.min(positions[:,1])        
    westSize = np.max(positions[:,1])    
    northSize = np.max(positions[:,0])
    southSize = np.min(positions[:,0])
    
    DeltaX = westSize - eastSize
    DeltaY = northSize - southSize
    objSize = [DeltaX + modelSize[2] , DeltaY + modelSize[1]]
    objSize[0] = np.int(np.ceil(objSize[0])*1.1)
    objSize[1] = np.int(np.ceil(objSize[1])*1.1)

    finalObj = (np.random.rand(objSize[0],objSize[1]) + 1j * np.random.rand(objSize[0],objSize[1])).astype(np.complex64)
    
    if centerPositions:
        
        newPositions = positions    
        newPositions[:,0] = newPositions[:,0] + DeltaX/2 - westSize
        newPositions[:,1] = newPositions[:,1] + DeltaY/2 - northSize
        
        return newPositions, finalObj
    else:
        return finalObj

###############################################################################    
    
    
def listROI(positions,roiSize,objSize):
    
    npos = np.shape(positions)[0]
    ROI = []    
    
    for cont in range(npos):
        roiMinY = positions[cont,1] - roiSize[1]/2 + objSize[0]/2
        roiMaxY = positions[cont,1] + roiSize[1]/2 + objSize[0]/2
        roiMinX = positions[cont,0] - roiSize[2]/2 + objSize[1]/2
        roiMaxX = positions[cont,0] + roiSize[2]/2 + objSize[1]/2
        
        ROI.append([roiMinY,roiMaxY,roiMinX,roiMaxX])
        
    return ROI
        
###############################################################################
        
def ptychoFFTShift(DifPads):

	newDifPads = []
    
	for Pad in DifPads:
		newDifPads.append( np.fft.fftshift(Pad) )

	return newDifPads

