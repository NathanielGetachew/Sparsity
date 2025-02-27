import tensorflow as tf
from tensorflow import keras
from keras import layers
from dropout import dropout  # Import custom dropout

def build_model(image_size=(150, 150), dropout_rate=0.5):
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=image_size + (3,)))
    
    # Convolutional layers with custom dropout
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Lambda(lambda x: dropout(x, dropout_rate, training=True)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Lambda(lambda x: dropout(x, dropout_rate, training=True)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Lambda(lambda x: dropout(x, dropout_rate, training=True)))
    
    # Flatten and dense layers with custom dropout
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Lambda(lambda x: dropout(x, dropout_rate, training=True)))
    
    # Output layer
    model.add(layers.Dense(6, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
