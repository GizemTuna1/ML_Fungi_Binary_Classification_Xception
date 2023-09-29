# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:42:59 2023

@author: gizem
"""

from keras.models import model_from_json
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception
import numpy as np

# Load the json file that contains the model's structure
model_path = "C:/Users/gizem/Masaüstü/Final Version/Model/model_structure.json"
weights_path = "C:/Users/gizem/Masaüstü/Final Version/Model/model_weights.h5"

with open(model_path, "r") as json_file:
    model_structure = json_file.read()
    model = model_from_json(model_structure)
    
model.load_weights(weights_path)

# Set the path to your dataset directory
dataset_dir = "C:/Users/gizem/Masaüstü/Final Version/Final Datasets/Test"

# Get a list of image file paths in the dataset directory
image_paths = list(Path(dataset_dir).glob("*.jpg"))

# Load the pre-trained Xception model for feature extraction
feature_extraction_model = xception.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Create an empty list to store predictions from each image
predictions = []

# Loop over each image path, load and preprocess the images, and make predictions
for img_path in image_paths:
    img = image.load_img(img_path, target_size=(299, 299))

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add a fourth dimension to the image (since Keras expects a batch of images, not a single image)
    images = np.expand_dims(image_array, axis=0)

    # Preprocess the input data specific to Xception
    images = xception.preprocess_input(images)

    # Use the pre-trained Xception model to extract features from the test image
    features = feature_extraction_model.predict(images)

    # Flatten the features to a 1D array before passing to the loaded model
    flattened_features = features.reshape(1, -1)

    # Given the flattened features, make a final prediction using the loaded model
    result = model.predict(flattened_features)

    # Since we are only testing one image with a possible class, we only need to check the first result's first element
    single_result = result[0][0]

    # Append the prediction to the list
    predictions.append(single_result)

# Convert the list of predictions to a numpy array for this model
predictions = np.array(predictions)
    
# Print the results for each image
for img_path, prediction in zip(image_paths, predictions):
    print("Image: {}, Likelihood that this image contains Aspergillus Fumigatus: {:.2f}%".format(img_path, int(prediction * 100)))

# Save the results to a file
output_directory = "C:/Users/gizem/Masaüstü/Final Version/Performance Metrics"
output_file = Path(output_directory) / "predictions.txt"
with open(output_file, "w") as file:
    for img_path, prediction in zip(image_paths, predictions):
        likelihood_percentage = prediction * 100
        file.write("Image: {}, Likelihood that this image contains Aspergillus Fumigatus: {:.2f}%\n".format(img_path, likelihood_percentage))


#Preparating for Analysis - Generating csv file containing actual labels:
import os
import csv

# Path to Unseen dataset directory
unseen_dataset_dir = "C:/Users/gizem/Masaüstü/Final Version/Final Datasets/Test"

# Path to save the CSV file
csv_file_path = "C:/Users/gizem/Masaüstü/Final Version/Performance Metrics/unseen_labels.csv"

# Open the CSV file for writing
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['Image', 'Label'])  # Image filename and Label columns

    # Iterate through image files in the directory
    for image_filename in os.listdir(unseen_dataset_dir):
        if image_filename.endswith('.jpg'):
            # Determine the label based on whether the image contains AFU or not
            if 'AFU' in image_filename:
                label = 1  # Image contains AFU
            else:
                label = 0  # Image does not contain AFU

            # Write the row to the CSV file
            csv_writer.writerow([image_filename, label])


#ANALYSIS OF PERFORMANCE METRICS:
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import csv

# Convert the list of ensemble predictions to a numpy array
predictions = np.array(predictions)

optimal_threshold = 0.90 #any prediction with a probability greater than or equal to 95% will be considered a positive prediction

# Apply the threshold to get final binary predictions
binary_predictions = [1 if pred >= optimal_threshold else 0 for pred in predictions]

# Create an empty list to store actual labels
actual_labels = []

# Load actual labels from the CSV file
labels_csv_file = "C:/Users/gizem/Masaüstü/Final Version/Performance Metrics/unseen_labels.csv"
with open(labels_csv_file, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        label = int(row[1])
        actual_labels.append(label)
        
# Calculate metrics
accuracy = accuracy_score(actual_labels, binary_predictions)
precision = precision_score(actual_labels, binary_predictions)
recall = recall_score(actual_labels, binary_predictions)
f1 = f1_score(actual_labels, binary_predictions)
conf_matrix = confusion_matrix(actual_labels, binary_predictions)

#Visualize:
    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define class labels
class_labels = ["Not AFU", "AFU"]

# Create a heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save the plot to your directory
plot_path = "C:/Users/gizem/Masaüstü/Final Version/Performance Metrics/confusion_matrix_plot.png"  
plt.savefig(plot_path)

# Display the plot
plt.show()

#Creating bar plot for metrics:
    
import matplotlib.pyplot as plt

# Define metric names and values
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metric_values = [accuracy, precision, recall, f1]

# Define colors for each metric
colors = ['blue', 'green', 'orange', 'red']

# Create a horizontal bar chart with annotations
plt.figure(figsize=(10, 6))
bars = plt.barh(metric_names, metric_values, color=colors, alpha=0.8)

# Add annotations on the bars
for metric_name, metric_value in zip(metric_names, metric_values):
    plt.text(metric_value + 0.02, metric_name, f'{metric_value:.3f}', va='center')

plt.xlim(0, 1)  # Set the x-axis limit to match the range of metric values
plt.xlabel("Values")
plt.title("Model Performance Metrics")

plot_path = "C:/Users/gizem/Masaüstü/Final Version/Performance Metrics/model_performance_metrics.png"  
plt.savefig(plot_path)
plt.show()

#ROC curve:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(actual_labels, binary_predictions)

# Calculate AUC-ROC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plot_path = "C:/Users/gizem/Masaüstü/Final Version/Performance Metrics/ROC_curve.png"  
plt.savefig(plot_path)
plt.show()




"""
# Threshold Exploration

import numpy as np
# Calculate G-Mean
gmeans = np.sqrt(tpr * (1 - fpr))
# Find index of the maximum G-Mean value
ix = np.argmax(gmeans)
# Get the corresponding threshold and G-Mean value
best_threshold = thresholds[ix]
best_gmean = gmeans[ix]
print('Best Threshold = %f, G-Mean = %.3f' % (best_threshold, best_gmean)) #Best Threshold = 1.000000, G-Mean = 0.981

OR

threshold_values = np.linspace(0.1, 0.9, num=9)

# Create dictionaries to store evaluation metrics for each threshold
precision_scores = {}
recall_scores = {}
f1_scores = {}

# Loop through each threshold value
for threshold in threshold_values:
    # Convert predictions to binary using the current threshold
    binary_predictions = [0 if pred < threshold else 1 for pred in predictions]

    # Calculate evaluation metrics
    precision = precision_score(actual_labels, binary_predictions)
    recall = recall_score(actual_labels, binary_predictions)
    f1 = f1_score(actual_labels, binary_predictions)
    
    # Store metrics in dictionaries
    precision_scores[threshold] = precision
    recall_scores[threshold] = recall
    f1_scores[threshold] = f1

# Find the threshold that maximizes the F1-score
optimal_threshold = max(f1_scores, key=f1_scores.get)

# Apply the optimal threshold to get final binary predictions
final_binary_predictions = [0 if pred < optimal_threshold else 1 for pred in predictions]

# Print the optimal threshold and evaluation metrics
print("Optimal Threshold:", optimal_threshold)
print("Precision:", precision_scores[optimal_threshold])
print("Recall:", recall_scores[optimal_threshold])
print("F1-Score:", f1_scores[optimal_threshold])

# Creating bar plot for metrics
import matplotlib.pyplot as plt

# ... (Remaining code)
"""


