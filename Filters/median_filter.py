import cv2
img=cv2.imread('/home/reynash/Downloads/download.jpeg')
median_filter=cv2.medianBlur(img,5)
cv2.imshow("Median Filter",median_filter)
cv2.waitKey(0)
