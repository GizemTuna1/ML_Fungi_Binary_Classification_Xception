# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:50:24 2023

@author: gizem
"""

# Import necessary libraries
from pathlib import Path
import numpy as np
import joblib
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
from PIL import Image

validation_path = Path("./Validation")

# Lists to store images and labels
images = []
labels = []

# Load all the AFU images from train dataset
for img_path in validation_path.glob("AFU/*.jpg"):
    try:
        img = Image.open(img_path)
        img = img.resize((299, 299))
        image_array = image.img_to_array(img)
        images.append(image_array)
        labels.append(1)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Load all the not-AFU images from train dataset
for img_path in validation_path.glob("Not-AFU/*.jpg"):
    try:
        img = Image.open(img_path)
        img = img.resize((299, 299))
        image_array = image.img_to_array(img)
        images.append(image_array)
        labels.append(0)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

# Convert lists to arrays
x_train = np.array(images)
y_train = np.array(labels)

# Preprocess input data specific to Xception
x_train = xception.preprocess_input(x_train)

# Load a pre-trained neural network (Xception) to use as a feature extractor
pretrained_nn = xception.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Specify the path to the existing "Extracted Features" folder
folder_path = "./Extracted_Features"

# Save the array of extracted features to a file
joblib.dump(features_x, os.path.join(folder_path, "x_validation.dat"))

# Save the matching array of expected values to a file
joblib.dump(y_train,os.path.join(folder_path, "y_validation.dat"))