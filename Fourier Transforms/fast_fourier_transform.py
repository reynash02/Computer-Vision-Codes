import cv2
import numpy as np
img=cv2.imread('/home/reynash/Downloads/download.jpeg')
f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
magnitude_spectrum=np.abs(fshift)
magnitude_spectrum=np.log(magnitude_spectrum+1)
cv2.imshow("Original Image",img)
cv2.imshow("Magnitude Spectrum",magnitude_spectrum)
cv2.waitKey(0)
