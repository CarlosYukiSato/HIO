# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:56:18 2017

@author: carlos.sato
"""
###############################################################################
import cv2
import os
import numpy as np
import scipy

from scipy.fftpack import fftn, ifftn

def ViewImage(Img, mult):
	np.savetxt("hioph.txt", Img)

im = (np.loadtxt('HDR/PHPuro.txt', delimiter=' '))[499-256:499+256,683-256:683+256]
im[im<0] = 0
DifPad = np.sqrt(im)

Mask = np.zeros(DifPad.shape,dtype=np.bool)
Mask[200:254,200:254] = True

###############################################################################

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.gpuarray import if_positive as cuif_positive
import skcuda.fft as cu_fft
import skcuda
from time import time

curtime = time()

iterations = 5000
beta = 0.9

phase_angle = np.random.rand(DifPad.shape[0],DifPad.shape[1]).astype(np.float32)*2.0*np.pi

#Define initial k, r space
initial_k = scipy.fftpack.ifftshift(DifPad).astype(np.complex64)
k_space = initial_k * np.exp(1j*phase_angle)

DifPad_gpu = gpuarray.to_gpu(initial_k)
k_space_gpu = gpuarray.to_gpu(k_space)

plan_forward = cu_fft.Plan(DifPad.shape, np.complex64, np.complex64)
plan_inverse = cu_fft.Plan(DifPad.shape, np.complex64, np.complex64)

r_space_gpu = gpuarray.zeros(DifPad.shape, np.complex64)
buffer_r_space = gpuarray.zeros(DifPad.shape, np.complex64)
cu_fft.ifft(k_space_gpu, buffer_r_space, plan_inverse, True)
MaskBoolean = gpuarray.to_gpu(Mask)

sample = gpuarray.zeros(k_space.shape, np.complex64)

RfacF = np.zeros((iterations,1)).astype(np.complex64);  
counter1=0; 
errorF=1;

iter = 1

buffer_r_space = buffer_r_space.real.astype(np.complex64)

print('Init time: ' + str(time()-curtime) + 's')
curtime = time()

while (iter < iterations):
    
    cu_fft.ifft(k_space_gpu, r_space_gpu, plan_inverse, True)
    r_space_gpu = r_space_gpu.real.astype(np.complex64)
    
    sample = pycuda.gpuarray.if_positive(MaskBoolean, r_space_gpu, sample)
    r_space_gpu = buffer_r_space - beta * r_space_gpu
    
    sample = pycuda.gpuarray.if_positive((sample.real < 0).astype(np.bool), r_space_gpu, sample)
    r_space_gpu = pycuda.gpuarray.if_positive(MaskBoolean, sample, r_space_gpu)
    
    buffer_r_space = r_space_gpu + 0.0;
    
    cu_fft.fft(r_space_gpu, k_space_gpu, plan_forward) 
    k_space_gpu = DifPad_gpu * (k_space_gpu / k_space_gpu.__abs__())
        
    iter = iter + 1

    if np.remainder(iter,500) == 0:
        print(iter)

print('Run in: ' + str(time()-curtime) + ' s')
ViewImage(buffer_r_space.get().real,1)

