image_size = 64

import numpy as np
import tensorflow as tf

# Load images
X = np.load("cell_images.npy")

# Normalize and add channel dimension
x_all = X.reshape(-1, image_size, image_size, 1).astype("float32") / 255.0

# Build bottlenecked, morphology-focused autoencoder
input_img = tf.keras.Input(shape=(image_size, image_size, 1))

# Much smaller latent space to force discrete-ish structure
latent_dim = 3

layers = tf.keras.layers
Model = tf.keras.Model

# Encoder: slightly lighter and with strong downsampling
x = input_img  # 64x64x1
for filters in [16, 32, 64, 128]:
    x = layers.Conv2D(
        filters,
        8,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

# Expect ~4x4 spatial size here
shape_before_flattening = x.shape[1:]
x = layers.Flatten()(x)

# Extra regularization to encourage more discrete structure in latent space
x = layers.Dropout(0.3)(x)

# Latent with stronger regularization to discourage overuse of dimensions
encoded = layers.Dense(
    latent_dim,
    name="encoded_morph",
    activity_regularizer=tf.keras.regularizers.l2(1e-3),
)(x)

# Decoder: mirror encoder capacity (but still lighter than main autoencoder)
x = layers.Dense(np.prod(shape_before_flattening), activation="relu")(encoded)
x = layers.Reshape(shape_before_flattening)(x)

for filters in [128, 64, 32, 16]:
    x = layers.Conv2DTranspose(
        filters,
        3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

decoded = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
)

# Simple train/val split
batch_size = 128
epochs = 200

num_val = int(0.1 * len(x_all))
x_val = x_all[:num_val]
x_train = x_all[num_val:]


def make_ds(data, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((data, data))
    if training:
        ds = ds.shuffle(buffer_size=len(data))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(x_train, training=True)
val_ds = make_ds(x_val, training=False)

# Callbacks
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

print("Training morphology-focused bottleneck autoencoder...")
history = autoencoder.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[reduce_lr, early_stop],
    verbose=1,
)

recon_loss = autoencoder.evaluate(x_all, x_all, verbose=0)
print(f"Reconstruction loss on full data (morph AE): {recon_loss:.4f}")

# Extract encoder and save morphology-focused latent space
encoder = Model(input_img, encoded)
encoded_morph = encoder.predict(x_all)

np.save("encoded_morph.npy", encoded_morph)
