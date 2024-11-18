import cv2
import numpy as np
from scipy.ndimage import generic_filter
img=cv2.imread('/home/reynash/Downloads/download.jpeg',cv2.IMREAD_GRAYSCALE)

def average_neighbour(val):
    return np.mean(val)

def lowest_neighbour(val):
    return np.min(val)

average_image=generic_filter(img,average_neighbour,size=(5,5))
lowest_image=generic_filter(img,lowest_neighbour,size=(5,5))

cv2.imshow("Average",average_image)
cv2.imshow("Lowest",lowest_image)

cv2.waitKey(0)