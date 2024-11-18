import cv2
import numpy as np
img=cv2.imread('/home/reynash/Downloads/download.jpeg')
neg2=cv2.bitwise_not(img)
stack=np.hstack((img,neg,neg2))
cv2.imshow("Image",stack)
cv2.waitKey(0)