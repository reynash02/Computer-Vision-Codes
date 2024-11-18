import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('/home/reynash/Downloads/download.jpeg')
hist=cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.title("Histogram")
plt.show()


gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
equalized=cv2.equalizeHist(gray_img)
cv2.imshow("Image",equalized)
cv2.waitKey(0)