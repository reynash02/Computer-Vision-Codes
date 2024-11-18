import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import selectivesearch

# Load pre-trained VGG16 model for feature extraction (convolutional layers only)
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Add fully connected layers to classify the extracted features
def build_classifier(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1000, activation='softmax')  # 1000 classes as in ImageNet
    ])
    return model

# Function to compute CNN features for a region
def compute_cnn_features(region, model):
    resized_region = resize(region, (224, 224))
    resized_region = np.expand_dims(resized_region, axis=0)
    resized_region = tf.keras.applications.vgg16.preprocess_input(resized_region)
    features = model.predict(resized_region)
    return features

# Function to classify regions using the fully connected layers
def classify_region(features, classifier):
    prediction = classifier.predict(features)
    decoded_predictions = tf.keras.applications.vgg16.decode_predictions(prediction, top=1)
    return decoded_predictions[0]

# Load image
image_path = '/home/reynash/Downloads/download.jpeg'
image = io.imread(image_path)

# Extract region proposals using Selective Search
def extract_region_proposals(image):
    img_lbl, regions = selectivesearch.selective_search(image, scale=400, sigma=0.8, min_size=50)
    candidates = []
    
    for r in regions:
        x, y, w, h = r['rect']

        # Filter regions based on size and aspect ratio
        if w < 30 or h < 30:  # Skip small regions
            continue
        if w / h > 2 or h / w > 2:  # Skip regions with odd aspect ratios
            continue
        candidates.append(r['rect'])

    return candidates

# Apply selective search to get region proposals
rects = extract_region_proposals(image)

# Limit the number of bounding boxes to 6-7
max_boxes = 7
if len(rects) > max_boxes:
    rects = rects[:max_boxes]  # Keep only the top 7 proposals

# Build the classifier for the extracted features
classifier = build_classifier((7, 7, 512))  # Feature map shape from VGG16

# Visualize proposed regions on the image
fig, ax = plt.subplots()
ax.imshow(image)

for (x, y, w, h) in rects:
    # Extract the region from the image
    region = image[y:y+h, x:x+w]

    # Compute CNN features
    features = compute_cnn_features(region, base_model)

    # Classify the region using the fully connected classifier
    prediction = classify_region(features, classifier)
    print(f"Predicted: {prediction}")

    # Draw rectangle on the image for the proposed region
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

plt.show()
