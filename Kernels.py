
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from functools import partial

kernelSource = -1
kernelsize = -1
blocksize=(256,1,1)
gridsize=(16,1)
numthreads = blocksize[0]*gridsize[0]

def ReplaceInKernel(textlist,text,header,string,offset=-1):
	index = text.find(header)
	if index==-1:
		raise header + ' not in kernel'
	index += len(header) + offset
	
	textlist[index] = string

def LoadKernel(filename,ksize,replace=[]):	
	kfile = open(filename)
	sourcecode = kfile.read()
	kfile.close()
	
	if len(replace) > 0:
		textlist = list(sourcecode)
		
		for k in range(0,len(replace),2):
			ReplaceInKernel(textlist,sourcecode,replace[k],replace[k+1])
		
		sourcecode = ''.join(str(letter) for letter in textlist)
		
	global kernelsize
	global kernelSource
	kernelsize = ksize
	kernelSource = SourceModule(sourcecode)

def GetFunction(funcname):
	global kernelSource
	global kernelsize
	global blocksize
	global gridsize
	
	Func = partial(kernelSource.get_function(funcname), kernelsize, block=blocksize, grid=gridsize)
	return Func
