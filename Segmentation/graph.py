import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('/home/reynash/Downloads/download.jpeg')
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

mask=np.zeros(image.shape[:2],np.uint8)

# Create models for foreground and background
bgd_model=np.zeros((1,65),np.float64)
fgd_model=np.zeros((1,65),np.float64)

# Define the rectangle where we expect the foreground to be (x, y, width, height)
# Adjust the rectangle dimensions based on your image
rect=(50,50,image.shape[1]-100,image.shape[0]-100)
cv2.grabCut(image,mask,rect,bgd_model,fgd_model,5,cv2.GC_INIT_WITH_RECT)

# Modify the mask: Pixels with value 2 or 0 are background, 1 or 3 are foreground
mask_2=np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Apply the mask to the original image to segment the foreground
segmented_image=image_rgb*mask_2[:,:,np.newaxis]
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented Image (GrabCut)')

plt.show()
