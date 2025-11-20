image_size = 64

import numpy as np
import tensorflow as tf

X = np.load('cell_images.npy')

# Prepare data
x_train_cnn = X.reshape(-1, image_size, image_size, 1).astype("float32") / 255.0

# Build autoencoder
input_img = tf.keras.Input(shape=(image_size, image_size, 1))

latent_dim = 64

# Encoder: 4 resolution levels, 2 convs per level
layers = tf.keras.layers
Model = tf.keras.Model

x = input_img  # 64x64x1
for filters in [32, 64, 128, 256]:
    # Downsampling conv
    x = layers.Conv2D(
        filters,
        3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    # Extra conv at same resolution
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

# At this point spatial size is 4x4, channels=256
shape_before_flattening = x.shape[1:]  # (4, 4, 256)
x = layers.Flatten()(x)
encoded = layers.Dense(latent_dim, name="encoded")(x)  # Latent representation

# Decoder: dense -> reshape -> Conv2DTranspose upsampling with extra convs
x = layers.Dense(np.prod(shape_before_flattening), activation="relu")(encoded)
x = layers.Reshape(shape_before_flattening)(x)

for filters in [256, 128, 64, 32]:
    x = layers.Conv2DTranspose(
        filters,
        3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

decoded = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

# Compile autoencoder: use MSE loss and a smaller learning rate
autoencoder = Model(input_img, decoded)
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
)

# Training configuration
batch_size = 128
epochs = 200  # rely on early stopping

num_val = int(0.1 * len(x_train_cnn))
x_val = x_train_cnn[:num_val]
x_train = x_train_cnn[num_val:]

def make_train_ds(data):
    ds = tf.data.Dataset.from_tensor_slices((data, data))
    ds = ds.shuffle(buffer_size=len(data))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_val_ds(data):
    ds = tf.data.Dataset.from_tensor_slices((data, data))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_train_ds(x_train)
val_ds = make_val_ds(x_val)

# Callbacks for better convergence
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1,
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1,
)

# Train autoencoder
print("Training autoencoder (no heavy augmentation, MSE loss)...")
history = autoencoder.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[reduce_lr, early_stop],
    verbose=1,
)

# Evaluate reconstruction loss on full data
recon_loss = autoencoder.evaluate(x_train_cnn, x_train_cnn, verbose=0)
print(f"Reconstruction loss on full data: {recon_loss:.4f}")

# Extract encoder for feature extraction
encoder = Model(input_img, encoded)

# Get latent representations
encoded_features = encoder.predict(x_train_cnn)

np.save("encoded.npy", encoded_features)