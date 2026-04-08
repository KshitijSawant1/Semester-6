import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Load California Housing Dataset
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1,1)

# Normalize input features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Normalize target to 0–1 (for sigmoid compatibility)
y = (y - y.min()) / (y.max() - y.min())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

iterations = 100

# Case 1: Forward Only (No Backpropagation)
np.random.seed(1)
W_forward = np.random.rand(X_train.shape[1],1)
b_forward = np.zeros((1,1))

loss_forward = []

for i in range(iterations):
    z = np.dot(X_train, W_forward) + b_forward
    output = sigmoid(z)
    loss = mse_loss(y_train, output)
    loss_forward.append(loss)

# Plot Forward Only
plt.figure()
plt.plot(loss_forward)
plt.title("Loss (Forward Only)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Case 2: Forward + Backpropagation
np.random.seed(1)
W_backward = np.random.rand(X_train.shape[1],1)
b_backward = np.zeros((1,1))

learning_rate = 0.01
loss_backward = []

for i in range(iterations):

    # Forward
    z = np.dot(X_train, W_backward) + b_backward
    output = sigmoid(z)

    # Loss
    loss = mse_loss(y_train, output)
    loss_backward.append(loss)

    # Backpropagation
    error = y_train - output
    d_output = error * sigmoid_derivative(output)

    W_backward += np.dot(X_train.T, d_output) * learning_rate
    b_backward += np.sum(d_output, axis=0, keepdims=True) * learning_rate

# Plot Backpropagation Only
plt.figure()
plt.plot(loss_backward)
plt.title("Loss (Forward + Backpropagation)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Combined Comparison (Subplot)
plt.figure()
plt.plot(loss_forward, label="Forward Only")
plt.plot(loss_backward, label="Forward + Backprop")
plt.title("Comparison of Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Final Loss Comparison
print("Final Loss without Backpropagation:", loss_forward[-1])
print("Final Loss with Backpropagation:", loss_backward[-1])
