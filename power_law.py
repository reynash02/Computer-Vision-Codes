import cv2
import numpy as np
img=cv2.imread('/home/reynash/Downloads/download.jpeg',cv2.IMREAD_GRAYSCALE)
gamma=2
normalized=img/255.0
gamma_correction=np.power(normalized,gamma)
result=np.uint8(gamma_correction*255)
stack=np.hstack((img,result))
cv2.imshow("Image",stack)
cv2.waitKey(0)