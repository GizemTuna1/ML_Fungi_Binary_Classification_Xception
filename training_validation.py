# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 17:38:27 2023
@author: gizem
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# Load data sets
x_train = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_features_augmented/x_train.dat")
y_train = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_features_augmented/y_train.dat")

x_val = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_features_augmented/x_val.dat")
y_val = joblib.load("C:/Users/gizem/Masaüstü/Final Version/Extracted_features_augmented/y_val.dat")

# Reshape the input data to 2-dimensional format (assuming the input is image data)
num_samples_train, img_height_train, img_width_train, num_channels_train = x_train.shape
x_train_reshaped = x_train.reshape(num_samples_train, img_height_train * img_width_train * num_channels_train)

x_val_reshaped = x_val.reshape(x_val.shape[0], -1)
# Data normalization
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train_reshaped)
x_val_scaled = scaler.transform(x_val_reshaped )  

# Create a model and add layers
model = Sequential()

model.add(Flatten(input_shape=x_train_scaled.shape[1:]))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Compile the model with a custom learning rate
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    metrics=['accuracy']
)

#Results:Best Learning Rate: 0.0001, Best Epochs: 50, Best Batch Size: 128, Best Accuracy: 0.9871
# Train the model
best_epochs = 50
best_batch_size = 64

history = model.fit(
    x_train_scaled,  # Use scaled data for training
    y_train,
    epochs=best_epochs,  # Use best_epochs here
    batch_size=best_batch_size,  # Use best_batch_size here
    shuffle=True,
    validation_data=(x_val_scaled, y_val)# Use scaled validation data here
)

#Analysis of accuracy:
# Get training and validation accuracies from the history
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Plot training and validation accuracies
plt.figure(figsize=(8, 6))
plt.plot(range(1, best_epochs + 1), training_accuracy, label='Training Accuracy')
plt.plot(range(1, best_epochs + 1), validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies')
plt.legend()

# Save the plot as an image file
save_dir = Path("C:/Users/gizem/Masaüstü/Final Version/Training Accuracy")
plt.savefig(save_dir / "accuracy_plot.png")

plt.show()

# Calculate accuracy gap
accuracy_gap = [train_acc - val_acc for train_acc, val_acc in zip(training_accuracy, validation_accuracy)]

# Print accuracy gap for each epoch
for epoch, gap in enumerate(accuracy_gap, start=1):
    print(f"Epoch {epoch}: Accuracy Gap = {gap:.4f}")
    
    
# Plot accuracy gap values
plt.figure(figsize=(8, 6))
plt.plot(range(1, best_epochs + 1), accuracy_gap, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Gap')
plt.title('Accuracy Gap Over Epochs')
plt.xticks(range(1, best_epochs + 1))
plt.grid(True)

# Save the plot as an image file
save_dir = Path("C:/Users/gizem/Masaüstü/Final Version/Training Accuracy")
plt.savefig(save_dir / "accuracy_gap_graph.png")

plt.show()

#Analysis of Loss:
# Get training and validation losses from the history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot training and validation losses
plt.figure(figsize=(8, 6))
plt.plot(range(1, best_epochs + 1), training_loss, label='Training Loss')
plt.plot(range(1, best_epochs + 1), validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)

# Save the plot as an image file
save_dir = Path("C:/Users/gizem/Masaüstü/Final Version/Training Accuracy")
plt.savefig(save_dir / "loss_plot.png")

plt.show()

# Calculate loss gap
loss_gap = [train_loss - val_loss for train_loss, val_loss in zip(training_loss, validation_loss)]

# Print accuracy gap for each epoch
for epoch, gap in enumerate(loss_gap, start=1):
    print(f"Epoch {epoch}: Loss Gap = {gap:.4f}")
    
# Plot loss gap values
plt.figure(figsize=(8, 6))
plt.plot(range(1, best_epochs + 1), loss_gap, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss Gap')
plt.title('Loss Gap Over Epochs')
plt.xticks(range(1, best_epochs + 1))
plt.grid(True)
plt.show()

# Specify the directory path
directory_path = r'C:/Users/gizem/Masaüstü/Final Version/Model'

# Create a Path object for the directory
save_dir = Path(directory_path)

# Save neural network structure
model_structure = model.to_json()
with open(save_dir / "model_structure.json", "w") as json_file:
    json_file.write(model_structure)

# Save neural network's trained weights
model.save_weights(save_dir / "model_weights.h5")
