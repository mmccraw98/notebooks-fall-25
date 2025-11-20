import numpy as np
import tensorflow as tf
from tensorflow import keras

cell_images = np.load('cell_images.npy')
species_labels = np.load('species_labels.npy')
basic_labels = np.load('basic_labels.npy')
order_labels = np.load('order_labels.npy')
major_clade_labels = np.load('major_clade_labels.npy')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cell_images, major_clade_labels, test_size=0.2, random_state=42)


image_size = 64
from tensorflow.keras import layers
n_labels = np.union1d(np.unique(y_train), np.unique(y_test)).size

# Prepare data for CNN (needs to be reshaped to include channel dimension)
x_train_cnn = X_train.reshape(-1, image_size, image_size, 1).astype('float32') / 255
x_test_cnn = X_test.reshape(-1, image_size, image_size, 1).astype('float32') / 255

# Convert labels to categorical (one-hot encoding)
y_train_cat = keras.utils.to_categorical(y_train, n_labels)
y_test_cat = keras.utils.to_categorical(y_test, n_labels)

# Build the CNN model
model = keras.Sequential([
    # First conv block
    layers.Conv2D(32, 6, activation='relu', padding='same', input_shape=(image_size, image_size, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, 6, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((4, 4)),
    layers.Dropout(0.25),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(n_labels, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

history = model.fit(
    x_train_cnn, 
    y_train_cat,
    batch_size=128,
    epochs=50,  # More epochs with early stopping
    validation_data=(x_test_cnn, y_test_cat),
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.imshow(confusion_matrix(y_test, np.argmax(model.predict(x_test_cnn), axis=1)))
plt.savefig('confusion_matrix_major_clade.png')
plt.show()

print(model.predict(x_test_cnn))