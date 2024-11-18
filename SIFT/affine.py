import cv2
import numpy as np
import matplotlib.pyplot as plt

def affine_transform(image,angle,scale):
    rows,cols=image.shape
    rotation_matrix=cv2.getRotationMatrix2D((cols/2,rows/2),angle,scale)
    return cv2.warpAffine(image,rotation_matrix,(cols, rows))

image=cv2.imread('/home/reynash/Downloads/download.jpeg', cv2.IMREAD_GRAYSCALE)
sift=cv2.SIFT_create()
scales=[0.8,1.0,1.2]
angles=[-30,0,30]

# List to store keypoints and descriptors
all_keypoints=[]
all_descriptors=[]
for scale in scales:
    for angle in angles:
        transformed_image=affine_transform(image,angle,scale)
        keypoints, descriptors=sift.detectAndCompute(transformed_image,None)
        all_keypoints.extend(keypoints)
        if descriptors is not None:
            all_descriptors.extend(descriptors)

output_image=cv2.drawKeypoints(image,all_keypoints,None)

plt.imshow(output_image, cmap='gray')
plt.title('Simulated ASIFT Keypoints')
plt.show()
