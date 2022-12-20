import cv2
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set the path to the directory containing the training images
path = "C:\\Users\\user\\Desktop\\hello\\trainning_images"

# Create an empty list to store the training data
training_data = []

# Create an empty list to store the labels (i.e., the names of the people)
labels = []

# Create an empty list to store the names of the people
names = []

# Iterate over the subdirectories in the training image directory
for subdir in os.listdir(path):
    # Set the path to the current subdirectory
    subdir_path = os.path.join(path, subdir)

    # If the current subdirectory is not a directory, skip it
    if not os.path.isdir(subdir_path):
        continue

    # Get the name of the person from the subdirectory name
    name = subdir
    names.append(name)

    # Iterate over the training images in the current subdirectory
    for file in os.listdir(subdir_path):
        # Load the image, convert it to grayscale, and apply histogram equalization
        image = cv2.imread(os.path.join(subdir_path, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        # Augment the image by rotating it and scaling it
        rotated = cv2.rotate(equalized, cv2.ROTATE_90_CLOCKWISE)
        scaled = cv2.resize(rotated, (150, 150))

        # Flatten the image into a 1D array
        flattened = scaled.flatten()

        # Store the flattened image and label in the training data list
        training_data.append(flattened)
        labels.append(name)

# Scale the training data using StandardScaler
scaler = StandardScaler()
scaled_training_data = scaler.fit_transform(training_data)

# Convert the labels to a NumPy array
labels_array = np.array(labels)

# Create a face recognition model
model = cv2.face.EigenFaceRecognizer_create()

# Train the model on the scaled training data and labels
model.train(scaled_training_data, labels_array)

# Save the trained model to a file
model.save('trained_model.yml')

# Save the names of the people for use in face identification later on
with open('names.txt', 'w') as f:
    for name in names:
        f.write(name + '\n')
