# Experiment 05: Hebbian Learning Rule

import numpy as np

# ----------------------------------------
# 1. Define Input Patterns and Target Output
# ----------------------------------------

# Input patterns (bipolar inputs preferred: -1, 1)
X = np.array([
    [1,  1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

# Target outputs (pattern classification)
T = np.array([1, -1, -1, -1])  # AND logic (bipolar form)

n_samples, n_features = X.shape

print("Input Patterns:\n", X)
print("Target Output:\n", T)

# ----------------------------------------
# 2. Initialize Weights and Bias
# ----------------------------------------
W = np.zeros(n_features)
b = 0

print("\nInitial Weights:", W)
print("Initial Bias:", b)

# ----------------------------------------
# 3. Hebbian Learning Rule Training
# ----------------------------------------
# w = w + x * t
# b = b + t

for i in range(n_samples):
    x = X[i]
    t = T[i]

    W = W + x * t
    b = b + t

# ----------------------------------------
# 4. Display Final Weights
# ----------------------------------------
print("\nFinal Weights:", W)
print("Final Bias:", b)

# ----------------------------------------
# 5. Testing Phase
# ----------------------------------------
print("\nTesting Results:")
print("X1  X2  Target  Output")

for i in range(n_samples):
    x = X[i]
    t = T[i]

    # Activation function
    y_in = np.dot(W, x) + b

    # Sign function
    y = 1 if y_in > 0 else -1

    print(f"{x[0]:<3} {x[1]:<3} {t:<7} {y}")