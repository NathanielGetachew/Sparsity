import os
from utils import prepare_data

# Set the data directory
data_dir = '/home/natty/Sparsity/data'

# Define the number of images per category to use
images_per_category = 1000

# Prepare the dataset with the specified number of images per category
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = prepare_data(
    data_dir, images_per_category=images_per_category
)

# Check dataset dimensions
print(f"Training data: {train_images.shape[0]} samples, {train_images.shape[1:]} image shape")
print(f"Validation data: {val_images.shape[0]} samples, {val_images.shape[1:]} image shape")
print(f"Test data: {test_images.shape[0]} samples, {test_images.shape[1:]} image shape")

# Optionally, save the datasets to disk for later use
import numpy as np
np.save(os.path.join(data_dir, 'train_images.npy'), train_images)
np.save(os.path.join(data_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(data_dir, 'val_images.npy'), val_images)
np.save(os.path.join(data_dir, 'val_labels.npy'), val_labels)
np.save(os.path.join(data_dir, 'test_images.npy'), test_images)
np.save(os.path.join(data_dir, 'test_labels.npy'), test_labels)

print("Datasets saved successfully!")
