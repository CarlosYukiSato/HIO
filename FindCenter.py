#	@author: giovanni.baraldi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def grad(p):
	if abs(p) < 1.5:
		return np.sin(p)
	else:
		return p

def FindCenter(DifPad, bPlot):
	difx = DifPad.shape[1]//2	# Center of image
	dify = DifPad.shape[0]//2
	hsize = 64
	qsize = hsize//2

	# Extract the center only, because it usually behaves better
	Autocorrelation = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(DifPad[dify-hsize:dify+hsize,difx-hsize:difx+hsize])))
	Phase = np.angle(Autocorrelation)

	if bPlot == True:
		plt.subplot(1,2,1)
		plt.imshow(Phase,cmap='hsv')

	CenterX = 0
	CenterY = 0
	bRunning = True

	while bRunning:
		GradX = 0.0
		GradY = 0.0
		NumGrads = 0

		avggrad2 = 0.0
		numgrad2 = 0

		for j in range(qsize,3*qsize):	# Compute average grad^2 in image
			for i in range(qsize,3*qsize): # Center usually behaves better
				dx = grad(Phase[j,i] - Phase[j,i-1])
				dy = grad(Phase[j,i] - Phase[j-1,i])
				avggrad2 += dx*dx + dy*dy
				numgrad2 += 1

		avggrad2 /= numgrad2
		for j in range(qsize,3*qsize): # only use pixel gradient if grad2 is relatively small
			for i in range(qsize,3*qsize):
				dx = grad(Phase[j,i] - Phase[j,i-1])
				dy = grad(Phase[j,i] - Phase[j-1,i])
				grad2 = dx*dx + dy*dy
				if grad2 < avggrad2:
					GradX += dx
					GradY += dy
					NumGrads += 1
									# Grad/NumGrads = grad is the average gradient of the image
		ErrorY = -GradY/NumGrads * Phase.shape[0] * 0.5/np.pi	# if dX is a small error from the center, then
		ErrorX = -GradX/NumGrads * Phase.shape[1] * 0.5/np.pi	# for a set of N elements, Phase = -K*X (property of FTs)
									# So d(Phase) = grad = -X*dK
									# but since Kn = 2pi n / N
									# then dK = 2pi/N for each pixel increment, and
									###  X = grad * N / 2pi  ###
		ErrorY = int(ErrorY+0.5)
		ErrorX = int(ErrorX+0.5)

		CenterX += ErrorX
		CenterY += ErrorY
		DifPad = np.roll(DifPad,ErrorY,0)
		DifPad = np.roll(DifPad,ErrorX,1)

		Autocorrelation = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(DifPad[dify-hsize:dify+hsize,difx-hsize:difx+hsize])))
		Phase = np.angle(Autocorrelation)

		if ErrorX == 0 and ErrorY == 0:
			bRunning = False
			errx = -GradX/NumGrads * Phase.shape[1] * 0.5 /np.pi
			erry = -GradY/NumGrads * Phase.shape[0] * 0.5 /np.pi
			print('Err: 0.' + str(int(100*np.sqrt(errx**2+erry**2)+0.5)))

		if bPlot == True:
			Autocorrelation = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(DifPad[dify-hsize:dify+hsize,difx-hsize:difx+hsize])))
			PhaseNew = np.angle(Autocorrelation)

			plt.subplot(1,2,2)
			plt.imshow(PhaseNew,cmap='hsv')
			plt.tight_layout()
			plt.show()

		return CenterY,CenterX

def Example():
	filepath = 'HDR/Pinhole2.npy'
	DifPad = np.load(filepath).astype(np.complex64)
	CenterY,CenterX = FindCenter(DifPad,True)

	print('CenterX: ' + str(CenterX))
	print('CenterY: ' + str(CenterY))
