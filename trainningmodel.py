import cv2
import os
import numpy as np

# Set the path to the directory containing the training images
path = "C:\\Users\\user\\Desktop\\hello\\trainning_images"

# Create an empty list to store the training data
training_data = []

# Create an empty list to store the labels (i.e., the IDs of the people)
labels = []

# Create a dictionary to map names to IDs
name_id_dict = {}

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

    # If the name is not in the dictionary, add it and assign it a unique ID
    if name not in name_id_dict:
        name_id_dict[name] = len(name_id_dict)

    # Get the ID of the person using the dictionary
    id = name_id_dict[name]

    # Iterate over the training images in the current subdirectory
    for file in os.listdir(subdir_path):
        # Load the image and convert it to grayscale
        image = cv2.imread(os.path.join(subdir_path, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Store the image and label in the training data list
        training_data.append(gray)
        labels.append(id)

# Train the face recognition model using the training data and labels
model = cv2.face.LBPHFaceRecognizer_create()
model.train(training_data, np.array(labels))

# Save the trained model to a file
model.save('trained_model.yml')

# Save the names of the people and their corresponding ID dictionaries
# for use in face identification later on
with open('names.txt', 'w') as f:
    for name in names:
        f.write(name + '\n')

with open('name_id_dict.txt', 'w') as f:
    for name, id in name_id_dict.items():
        f.write(name + ' ' + str(id) + '\n')


