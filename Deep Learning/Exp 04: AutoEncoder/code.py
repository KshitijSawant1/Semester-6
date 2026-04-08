import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# ----------------------------
# 1) Load MNIST dataset
# ----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to (N, 28, 28, 1)
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print("Shape of training data:", x_train.shape)
print("Shape of testing data:", x_test.shape)

# ----------------------------
# 2) Define Simple Autoencoder
# ----------------------------
class SimpleAutoencoder(Model):
    def __init__(self, latent_dimensions):
        super(SimpleAutoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(latent_dimensions, activation="relu"),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(28 * 28, activation="sigmoid"),
            layers.Reshape((28, 28, 1)),
        ])

    def call(self, input_data):
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded

latent_dimensions = 64
autoencoder = SimpleAutoencoder(latent_dimensions)
autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

# ----------------------------
# 3) Train Autoencoder
# ----------------------------
history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# ----------------------------
# 4) Plot Loss Curve
# ----------------------------
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Autoencoder Loss Curve (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

# ----------------------------
# 5) Reconstruct test images
# ----------------------------
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# ----------------------------
# 6) Show digits 0 to 9 (one sample each)
#    Top row: Original, Bottom row: Reconstructed
# ----------------------------
indices_0_to_9 = []
for digit in range(10):
    idx = np.where(y_test == digit)[0][0]   # first occurrence of each digit
    indices_0_to_9.append(idx)

plt.figure(figsize=(18, 4))

# Original row
for i, idx in enumerate(indices_0_to_9):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Original {i}")
    plt.axis("off")

# Reconstructed row
for i, idx in enumerate(indices_0_to_9):
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Recon {i}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ----------------------------
# 7) Print Reconstruction MSE on test set
# ----------------------------
test_recon_mse = np.mean((x_test - decoded_imgs) ** 2)
print("Reconstruction MSE on test set:", test_recon_mse)