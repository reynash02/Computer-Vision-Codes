import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image=cv2.imread('/home/reynash/Downloads/download.jpeg')
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
pixel_values=image_rgb.reshape((-1,3))
pixel_values=np.float32(pixel_values)


# We stop either after 100 iterations or when the epsilon (accuracy) reaches 0.2
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2)
k = 3
kmeans=KMeans(n_clusters=k,random_state=42)
kmeans.fit(pixel_values)
labels=kmeans.labels_
centers=kmeans.cluster_centers_
centers=np.uint8(centers)
# Map the labels back to the original image's pixel values
segmented_image=centers[labels.flatten()]
segmented_image=segmented_image.reshape(image_rgb.shape)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f'Segmented Image with {k} clusters')

plt.show()
