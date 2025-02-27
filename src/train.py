import tensorflow as tf
from model import build_model
import numpy as np
from loss_plot import save_loss_plot  # Import the function

# Set the data directory
data_dir = '/home/natty/Sparsity/data'

train_images = np.load(f"{data_dir}/train_images.npy")
train_labels = np.load(f"{data_dir}/train_labels.npy")
val_images = np.load(f"{data_dir}/val_images.npy")
val_labels = np.load(f"{data_dir}/val_labels.npy")
test_images = np.load(f"{data_dir}/test_images.npy")
test_labels = np.load(f"{data_dir}/test_labels.npy")

# Reduce the batch size and image size
image_size = (100, 100)  # Reduced image size for memory efficiency
batch_size = 16  # Reduced batch size to avoid memory overload

# Data Augmentation using ImageDataGenerator to avoid loading all images into memory
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

# Create generators for training, validation, and test data
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=batch_size)
test_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)

# Build the model
dropout_rate = 0.5  # 50% dropout
model = build_model(image_size=image_size, dropout_rate=dropout_rate)

# Training the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    verbose=1
)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the plot of training and validation loss over epochs
save_loss_plot(history)
print("Loss and Accuracy plots saved as 'training_loss.png' and 'training_accuracy.png'")
