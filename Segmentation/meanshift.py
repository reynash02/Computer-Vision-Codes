import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('/home/reynash/Downloads/download.jpeg')
image_luv=cv2.cvtColor(image,cv2.COLOR_BGR2LUV)

# Apply mean shift filtering to smooth the image and segment it
# sp: spatial window radius, sr: color window radius
mean_shift_result=cv2.pyrMeanShiftFiltering(image_luv,sp=21,sr=51)
mean_shift_rgb=cv2.cvtColor(mean_shift_result,cv2.COLOR_LUV2RGB)
plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(mean_shift_rgb)
plt.title('Segmented Image (Mean Shift)')

plt.show()
