import cv2
img=cv2.imread('/home/reynash/Downloads/download.jpeg')
median_filter=cv2.GaussianBlur(img,(5,5),0)
cv2.imshow("Gaussian filter",median_filter)
cv2.waitKey(0)
