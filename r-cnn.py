import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np

# Function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, boxes):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Adjust box coordinates and color as needed
    for box in boxes:
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, color='red', fill=False, linewidth=2))

    plt.axis('off')
    plt.show()

# Create the computer vision model (assuming face classification)
def create_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Load the image
image_path = '/home/reynash/Downloads/download (1).jpeg'
preprocessed_image = preprocess_image(image_path)

# Create the model (replace num_classes with the number of face classes)
num_classes = 2  # Adjust based on your classification (e.g., face vs. non-face)
model = create_model(num_classes)

# Compile the model (optional)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (if necessary)
# model.fit(training_data, validation_data, epochs=10)

# Make predictions (assuming pre-trained for face classification)
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions)

# Print the predicted class (modify based on your classification)
if predicted_class == 0:
    print("Image likely contains a face")
else:
    print("Image likely does not contain a face")

# **Use a pre-trained object detection model (e.g., SSD MobileNet V2)**
object_detection_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
object_detections = object_detection_model.predict(preprocessed_image)

# Decode predictions using the object detection model's decoding function
face_boxes = tf.keras.applications.mobilenet_v2.decode_predictions(object_detections)[0]

# Extract face bounding boxes
face_boxes = [box[2] for box in face_boxes if 'face' in box[1]]

# Draw bounding boxes
original_image = np.array(Image.open(image_path))  # Load original image
draw_bounding_boxes(original_image.copy(), face_boxes)