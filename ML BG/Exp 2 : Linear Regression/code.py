# Experiment 02: Linear Regression using sklearn dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# ----------------------------------------
# 1. Load Dataset (Sklearn Built-in)
# ----------------------------------------
data = fetch_california_housing()

X = data.data[:, 0].reshape(-1, 1)  # Use one feature (MedInc)
y = data.target                     # Target (House Price)

print("Feature Name:", data.feature_names[0])

# ----------------------------------------
# 2. Manual Linear Regression Calculation
# ----------------------------------------
X_manual = X.flatten()

mean_x = np.mean(X_manual)
mean_y = np.mean(y)

# Calculate slope (B1)
numerator = np.sum((X_manual - mean_x) * (y - mean_y))
denominator = np.sum((X_manual - mean_x) ** 2)

B1 = numerator / denominator

# Calculate intercept (B0)
B0 = mean_y - B1 * mean_x

print("\nManual Calculation:")
print("Intercept (B0):", B0)
print("Slope (B1):", B1)

# Predictions (Manual)
y_pred_manual = B0 + B1 * X_manual

# ----------------------------------------
# 3. Using Scikit-learn Model
# ----------------------------------------
model = LinearRegression()
model.fit(X, y)

y_pred_sklearn = model.predict(X)

print("\nScikit-learn Model:")
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# ----------------------------------------
# 4. Visualization
# ----------------------------------------
plt.figure(figsize=(8,6))

# Scatter Plot
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')

# Manual Line
plt.plot(X, y_pred_manual, color='red', label='Manual Regression')

# Sklearn Line
plt.plot(X, y_pred_sklearn, color='green', linestyle='dashed', label='Sklearn Regression')

plt.title("Linear Regression (Sklearn Dataset)")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.legend()

# ----------------------------------------
# 5. Display Plot
# ----------------------------------------
plt.show()