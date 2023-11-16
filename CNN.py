# CNN.py
pip install opencv-python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread(r'C:\Users\Admin\Downloads\01\unnamed.jpg')

if image is not None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title('Displayed Image')
    plt.axis('off')  
    plt.show()
else:
    print("Error: Image not found or the path is incorrect")

height, width = image.shape[:2]
#print(image)

# Define the center of the image
center = (width // 2, height // 2)

# Define the rotation matrix
angle = 45 # Rotate image by 45 degrees
scale = 1.0 # No scaling
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Rotate the image
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display the rotated image
rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
plt.imshow(rotated_image_rgb)
plt.axis('off')
plt.show()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#print(gray_image)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Display the original and equalized images side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(gray_image, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Original Image')
ax[1].imshow(equalized_image, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Equalized Image')

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Detect edges using the Canny edge detector
edges = cv2.Canny(blurred_image, 50, 150)

# Display the edges
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()


#7. Image Segmentation
#Code: Adaptive Thresholding and Contour Detection
# Apply adaptive thresholding
adaptive_threshold = cv2.adaptiveThreshold(
gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#Find contours
contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_TREE,
cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
image_with_contours = image_rgb.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
# Display the image with contours
plt.imshow(image_with_contours)
plt.axis('off')
plt.show()

#8 Classification 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Define the CNN architecture
# Load CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential([
 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, (3, 3), activation='relu'),
 layers.Flatten(),
 layers.Dense(64, activation='relu'),
 layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
 metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10,
 validation_data=(test_images, test_labels))

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# Test the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
plt.subplot(1, 2, 2)
plt.title('Test Accuracy: {:.3f}'.format(test_acc))
plt.axis('off')
plt.show()



