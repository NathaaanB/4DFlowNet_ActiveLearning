import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.ndimage import zoom
import h5py

# -----------------------------
# Data preparation
# -----------------------------
HR_SHAPE = (32, 32, 32, 1)   # Depth x Height x Width x Channels
scale = 2
LR_SHAPE = (HR_SHAPE[0]//scale, HR_SHAPE[1]//scale, HR_SHAPE[2]//scale, 1)

def make_ellipsoid(shape, radius_scale=0.4):
    """Create a 3D ellipsoid as HR synthetic data."""
    z, y, x = np.indices(shape)
    center = np.array(shape) / 2
    radii = np.array(shape) * radius_scale / 2
    norm = ((z-center[0])**2 / radii[0]**2
          + (y-center[1])**2 / radii[1]**2
          + (x-center[2])**2 / radii[2]**2)
    return (norm <= 1).astype(np.float32)

def generate_dataset(n_samples=50):
    hr_vols = []
    for _ in range(n_samples):
        hr = make_ellipsoid(HR_SHAPE[:-1])
        hr = np.expand_dims(hr, -1)  # add channel
        hr_vols.append(hr)
    return np.array(hr_vols, dtype="float32")

def downsample_3d(volumes, scale=2):
    lr_vols = []
    for vol in volumes:
        vol = vol[..., 0]  # remove channel for zoom
        factors = (1/scale, 1/scale, 1/scale)  # downsample in D,H,W
        lr = zoom(vol, factors, order=1)  # linear interpolation
        lr = np.expand_dims(lr, -1)       # restore channel
        lr_vols.append(lr)
    return np.array(lr_vols, dtype="float32")

# Generate dataset
x_train = generate_dataset(5000)
x_test  = generate_dataset(100)

lr_train = downsample_3d(x_train, scale=scale)
lr_test  = downsample_3d(x_test, scale=scale)

print("HR train:", x_train.shape, "LR train:", lr_train.shape)

# -----------------------------
# Model definition (with dropout for epistemic)
# -----------------------------
def build_sr3d_model(lr_shape, dropout_rate=0.2):
    inputs = layers.Input(shape=lr_shape)

    x = layers.Conv3D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x, training=True)

    x = layers.UpSampling3D(size=(scale, scale, scale))(x)
    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)

    # Concatenate outputs: [mu, log_var] along channels
    out = layers.Conv3D(2, 3, padding="same")(x)

    return models.Model(inputs, out)

model = build_sr3d_model(LR_SHAPE)
model.summary()

# -----------------------------
# Loss function (heteroscedastic regression)
# -----------------------------
def heteroscedastic_loss(y_true, y_pred):
    mu = y_pred[..., 0:1]
    log_var = y_pred[..., 1:2]
    precision = tf.exp(-log_var)
    return tf.reduce_mean(precision * (y_true - mu)**2 + log_var)

model.compile(optimizer="adam", loss=heteroscedastic_loss)

# -----------------------------
# Training
# -----------------------------
history = model.fit(
    lr_train, x_train,
    validation_data=(lr_test, x_test),
    epochs=60,
    batch_size=20,
    verbose=1
)

# -----------------------------
# Monte Carlo Prediction
# -----------------------------
def mc_predict(model, x, T=16):
    mu_list, log_var_list = [], []
    for _ in range(T):
        y_out = model(x, training=True)
        mu = y_out[..., 0:1]
        log_var = y_out[..., 1:2]
        mu_list.append(mu.numpy())
        log_var_list.append(log_var.numpy())
    mu_mean = np.mean(mu_list, axis=0)
    aleatoric = np.mean(np.exp(log_var_list), axis=0)
    epistemic = np.var(mu_list, axis=0)
    total = aleatoric + epistemic
    return mu_mean, aleatoric, epistemic, total

mu_pred, alea, epi, total_var = mc_predict(model, lr_test, T=16)

# -----------------------------
# Save to HDF5
# -----------------------------
output_file = "sr3d_results_test.h5"

with h5py.File(output_file, "w") as f:
    f.create_dataset("lr", data=lr_test, compression="gzip")
    f.create_dataset("hr", data=x_test, compression="gzip")
    f.create_dataset("mu", data=mu_pred, compression="gzip")
    f.create_dataset("aleatoric", data=alea, compression="gzip")
    f.create_dataset("epistemic", data=epi, compression="gzip")
    f.create_dataset("total", data=total_var, compression="gzip")

print(f"✅ Saved results to {output_file}")
