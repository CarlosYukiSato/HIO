import numpy as np
import numpy.linalg as linalg

def GetMatLine(x,y):
	return [x**2, y**2, x*y, x, y, 1]
def MakeMat(neib):
	mat = []
	for j in range(-neib,neib+1):
		for i in range(-neib,neib+1):
			mat.append( GetMatLine(i,j) )
	return mat
def Solve(Fs,neib):
	Am = MakeMat(neib)
	At = np.transpose(Am)
	AtFS = np.matmul(At,Fs)
	invA = linalg.inv(np.matmul(At,Am))
	return np.matmul(invA,AtFS)
	
def GetMin2(Fs,neib):
	vmin = 1E10
	jmin = 0
	imin = 0
	for j in range(0,3):
		for i in range(0,3):
			val = Fs[3*j+i]
			if val < vmin:
				vmin = val+0
				jmin = j-1
				imin = i-1
	return imin,jmin
	
def GetMin(Fs,neib):
	Coef = Solve(Fs,neib)
	vmin = 1E10
	jmin = 0
	imin = 0
	for j in range(-neib,neib+1):
		for i in range(-neib,neib+1):
			val = Coef[0]*i**2 + Coef[1]*j**2 + Coef[2]*j*i + Coef[3]*i + Coef[4]*j + Coef[5]
			if val < vmin:
				vmin = val+0
				jmin = j
				imin = i
	return imin,jmin
	
import skcuda.fft as cu_fft

def RunCorrection(neib,ROI,DifPad,rspace,kspace,exitWave,buffer_exitWave,finalObj,offsetx,offsety,objsizex,roisizex,CopyFromROI,ExitwaveAndBuffer,ApplyDifPad,cufftplan,aperture,fcachevector):
	Fs = []		
	for jpos in range(-neib,neib+1):
		for ipos in range(-neib,neib+1):
			CopyFromROI(rspace, finalObj, np.int32(offsety+jpos), np.int32(offsetx+ipos), roisizex, objsizex)
			
			ExitwaveAndBuffer(exitWave, buffer_exitWave, aperture, rspace) # Compute exitwaves
			cu_fft.fft(exitWave,kspace,cufftplan) # kspace = wave at detector
			ApplyDifPad(kspace,DifPad,fcachevector) # replace amplitudes.
			cu_fft.ifft(kspace,exitWave,cufftplan,True)	# new exitwave
				
			errori = np.sum(((exitWave-buffer_exitWave).__abs__()**2).get())
			Fs.append(errori+0)
	return GetMin(Fs,neib)
			
						
			#if iter>200 and iter%60 == 0 and aper != 12:
		#	mini = 0
		#	minj = 0
		#	diffmin = np.sum(((exitWave-buffer_exitWave).__abs__()**2).get())
		#	jposesvar = [0,0,1,-1]#,1,-1,1,-1]
		#	iposesvar = [-1,1,0,0]#,1,1,-1,-1]
			
		#	for uselessvar in range(0,1):
		#		for uselessindex in range(0,4):
		#			jpos = jposesvar[uselessindex]
		#			ipos = iposesvar[uselessindex]
		#			CopyFromROI(rspace, finalObj, np.int32(offsety+jpos), np.int32(offsetx+ipos), roisizex, objsizex)
			
		#			ExitwaveAndBuffer(exitWave, buffer_exitWave, aperture, rspace) # Compute exitwaves
		#			cu_fft.fft(exitWave,kspace,cufftplan) # kspace = wave at detector
		#			ApplyDifPad(kspace,DifPads[aper],fcachevector) # replace amplitudes.
		#			cu_fft.ifft(kspace,exitWave,cufftplan,True)	# new exitwave
				
		#			errori = np.sum(((exitWave-buffer_exitWave).__abs__()**2).get())
		#			if errori < diffmin:
		#				diffmin = errori+0
		#				mini = ipos+0
		#				minj = jpos+0	
