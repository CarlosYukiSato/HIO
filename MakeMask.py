import numpy as np
import cv2
import os
from copy import deepcopy
from scipy import ndimage

Image = np.asarray(cv2.imread('Pinhole.png',cv2.IMREAD_GRAYSCALE),dtype=np.float32)
Mask = np.zeros(Image.shape,dtype=np.float32)
Mask[480-52:480+52,480-52:480+52] = 1
Image *= Mask

Avg = np.sum(Image)/104/104
Mask[Image<Avg] = 0

Mask = 255*ndimage.binary_dilation(Mask).astype(np.float32)
		
cv2.imwrite("PinholeMask.png", Mask)
Mask = cv2.imread("PinholeMask.png")
