import cv2
import numpy as np
img=cv2.imread('/home/reynash/Downloads/download.jpeg',cv2.IMREAD_GRAYSCALE)
img_float32=np.float32(img)
dft=cv2.dft(img_float32,flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift=np.fft.fftshift(dft)
magnitude_spectrum=cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])
magnitude_spectrum=np.log(magnitude_spectrum+1)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
cv2.waitKey(0)
