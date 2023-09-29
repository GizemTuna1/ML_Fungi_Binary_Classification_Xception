# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:14:11 2023

@author: gizem
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
#import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

# Load data
features_x = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_Features/x_train.dat")
y_train = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_Features/y_train.dat")

x_validation = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_Features/x_validation.dat")
y_validation = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_Features/y_validation.dat")

# Preprocess data
# Reshape the features to a 2D format
features_x_reshaped = features_x.reshape(features_x.shape[0], -1)
x_validation_reshaped = x_validation.reshape(x_validation.shape[0], -1)
# Create a MinMaxScaler instance and fit it on the training fold features
scaler = MinMaxScaler()
features_x_scaled = scaler.fit_transform(features_x.reshape(features_x.shape[0], -1))
x_validation_scaled = scaler.transform(x_validation.reshape(x_validation.shape[0], -1))

# Hyperparameters
learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [10, 20, 50]
batch_sizes = [32, 64, 128]

# Initialize best values and accuracy
best_accuracy = 0.0
best_learning_rate = None
best_epochs = None
best_batch_size = None

# Lists to store results
hyperparameter_results = []

# Specify the directory to save plots and table
save_dir = "C:/Users/gizem/Masaüstü/Final Version/Optimization"
plot_dir = os.path.join(save_dir, "plot_directory")
os.makedirs(plot_dir, exist_ok=True)

# Loop through each combination of hyperparameters
for learning_rate in learning_rates:
    for epochs in epochs_list:
        for batch_size in batch_sizes:
            print(f"Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}")

            # Create and compile model
            model = Sequential([
                Dense(256, activation='relu', input_shape=(features_x_scaled.shape[1],)),
                BatchNormalization(),
                Dropout(0.6),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Train model
            history = model.fit(features_x_scaled, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(x_validation_scaled, y_validation),
                                verbose=2) #providing an update at the end of each epoch
            
            # Extract accuracy and loss values
            training_accuracy = history.history['accuracy']
            validation_accuracy = history.history['val_accuracy']
            training_loss = history.history['loss']
            validation_loss = history.history['val_loss']

            # Append results to the list
            hyperparameter_results.append({
                'Learning Rate': learning_rate,
                'Epochs': epochs,
                'Batch Size': batch_size,
                'Training Accuracy': training_accuracy,
                'Validation Accuracy': validation_accuracy,
                'Training Loss': training_loss,
                'Validation Loss': validation_loss
            })

            # Save training and validation accuracy/loss plots
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(training_accuracy, label='Training Accuracy')
            plt.plot(validation_accuracy, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(training_loss, label='Training Loss')
            plt.plot(validation_loss, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.legend()

            plot_filename = f"lr_{learning_rate:.4f}_epochs_{epochs}_batch_{batch_size}_acc_loss.png"
            plot_path = os.path.join(plot_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()

# Create and export table
table_data = []
for result in hyperparameter_results:
    table_data.append([
        result['Learning Rate'],
        result['Epochs'],
        result['Batch Size'],
        result['Training Accuracy'][-1],
        result['Validation Accuracy'][-1],
        result['Training Loss'][-1],
        result['Validation Loss'][-1]
    ])

table_headers = ["Learning Rate", "Epochs", "Batch Size", "Train Acc", "Val Acc", "Train Loss", "Val Loss"]

table_str = tabulate(table_data, headers=table_headers, tablefmt="grid")
with open(os.path.join(save_dir, "hyperparameter_table.txt"), "w") as table_file:
    table_file.write(table_str)
#Convert to Excel file
import csv
# Specify the CSV file name
csv_file_name = 'hyperparameter_results.csv'

# Write data to CSV with formatting for decimal places
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Learning Rate', 'Epochs', 'Batch Size', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'])
    
    for row in table_data:
        formatted_row = [f'{value:.4f}' if isinstance(value, float) else value for value in row]
        writer.writerow(formatted_row)

print(f'Data saved to {csv_file_name}')



