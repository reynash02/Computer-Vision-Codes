import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_hog(img, cell_size=8, block_size=2, bins=9):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients in x and y direction
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    
    # Calculate magnitude and angle of gradients
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # Number of cells in x and y direction
    cell_x = gray.shape[1] // cell_size
    cell_y = gray.shape[0] // cell_size
    
    # Create HOG descriptor matrix
    hog_descriptor = np.zeros((cell_y, cell_x, bins))
    
    # Histogram bins based on angle range [0, 180)
    bin_width = 180 / bins
    
    # Calculate HOG for each cell
    for i in range(cell_y):
        for j in range(cell_x):
            # Extract magnitude and angle of gradients for the current cell
            cell_mag = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_angle = angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            
            # Create histogram for the cell
            for k in range(cell_mag.shape[0]):
                for l in range(cell_mag.shape[1]):
                    bin_idx = int(cell_angle[k, l] // bin_width) % bins
                    hog_descriptor[i, j, bin_idx] += cell_mag[k, l]
                    
    return hog_descriptor

def visualize_hog(hog_descriptor, cell_size=8, bins=9):
    h, w, _ = hog_descriptor.shape
    hog_image = np.zeros((h * cell_size, w * cell_size), dtype=np.uint8)
    bin_width = 180 / bins

    for i in range(h):
        for j in range(w):
            cell_center = (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)
            for b in range(bins):
                angle = b * bin_width + bin_width / 2
                rad = np.deg2rad(angle)
                dx = int(np.cos(rad) * hog_descriptor[i, j, b] * 0.5)
                dy = int(np.sin(rad) * hog_descriptor[i, j, b] * 0.5)
                cv2.line(hog_image, (cell_center[0] - dx, cell_center[1] - dy),
                         (cell_center[0] + dx, cell_center[1] + dy), 255, 1)
    
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Visualization")
    plt.show()

# Load image and compute HOG
image = cv2.imread("/home/reynash/Downloads/download.jpeg")
hog_descriptor = compute_hog(image)
visualize_hog(hog_descriptor)
