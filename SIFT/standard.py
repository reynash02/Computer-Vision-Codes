import cv2
import matplotlib.pyplot as plt

image=cv2.imread('/home/reynash/Downloads/download.jpeg',cv2.IMREAD_GRAYSCALE)
sift=cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints,descriptors=sift.detectAndCompute(image,None)
output_image=cv2.drawKeypoints(image,keypoints,None)
plt.imshow(output_image,cmap='gray')
plt.title('SIFT Keypoints')
plt.show()
