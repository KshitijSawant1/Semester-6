# Experiment 07: Single Layer Perceptron Model

import numpy as np

# ----------------------------------------
# 1. Define Dataset (Linearly Separable)
# Example: AND Gate (Bipolar form)
# ----------------------------------------

X = np.array([
    [1,  1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

# Target Output (Bipolar)
y = np.array([1, -1, -1, -1])

n_samples, n_features = X.shape

print("Input:\n", X)
print("Target:\n", y)

# ----------------------------------------
# 2. Initialize Weights and Bias
# ----------------------------------------
W = np.zeros(n_features)
b = 0
learning_rate = 0.1

# ----------------------------------------
# 3. Activation Function (Hard Limiter)
# ----------------------------------------
def activation(y_in):
    return 1 if y_in >= 0 else -1

# ----------------------------------------
# 4. Training Process
# ----------------------------------------
epochs = 10

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    
    for i in range(n_samples):
        x = X[i]
        target = y[i]
        
        # Linear combination
        y_in = np.dot(W, x) + b
        
        # Predicted output
        y_pred = activation(y_in)
        
        # Update rule
        if y_pred != target:
            W = W + learning_rate * target * x
            b = b + learning_rate * target
        
        print(f"Input: {x}, Predicted: {y_pred}, Target: {target}")

# ----------------------------------------
# 5. Final Weights
# ----------------------------------------
print("\nFinal Weights:", W)
print("Final Bias:", b)

# ----------------------------------------
# 6. Testing Phase
# ----------------------------------------
print("\nTesting Results:")
print("X1  X2  Target  Output")

for i in range(n_samples):
    y_in = np.dot(W, X[i]) + b
    y_out = activation(y_in)
    print(f"{X[i][0]:<3} {X[i][1]:<3} {y[i]:<7} {y_out}")