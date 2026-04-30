import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

class AE(Model):
    def __init__(self):
        super().__init__()
        self.enc = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(64,activation='relu')
        ])
        self.dec = tf.keras.Sequential([
            layers.Dense(28*28,activation='sigmoid'),
            layers.Reshape((28,28,1))
        ])
    def call(self,x):
        return self.dec(self.enc(x))

model = AE()
model.compile(optimizer='adam',loss='mse')

h = model.fit(x_train,x_train,epochs=5,validation_data=(x_test,x_test))

plt.plot(h.history['loss'],label='train')
plt.plot(h.history['val_loss'],label='val')
plt.legend(); plt.title("Loss"); plt.show()

decoded = model(x_test)

print("MSE:", np.mean((x_test-decoded)**2))