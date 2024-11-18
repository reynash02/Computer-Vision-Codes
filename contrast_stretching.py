import cv2
import numpy as np
img=cv2.imread('/home/reynash/Downloads/download.jpeg')
x=np.max(img)
y=np.min(img)
stretched=(((img-y)/(x-y))*255).astype(np.uint8)
cv2.imshow("Stretched image",stretched)
cv2.waitKey(0)