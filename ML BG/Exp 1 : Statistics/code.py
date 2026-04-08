# Experiment 01: Statistical Operations in Machine Learning

import numpy as np
print()
# Sample Data
X = np.array([2, 4, 6, 8, 10])
Y = np.array([1, 3, 5, 7, 9])

print("Dataset X:", X)
print("Dataset Y:", Y)

# ----------------------------------------
# 1. Mean
# ----------------------------------------
mean_X = np.mean(X)
mean_Y = np.mean(Y)

print("\nMean of X:", mean_X)
print("Mean of Y:", mean_Y)

# ----------------------------------------
# 2. Variance
# ----------------------------------------
var_X = np.var(X)
var_Y = np.var(Y)

print("\nVariance of X:", var_X)
print("Variance of Y:", var_Y)

# ----------------------------------------
# 3. Covariance
# ----------------------------------------
cov_matrix = np.cov(X, Y)

print("\nCovariance Matrix:\n", cov_matrix)

# ----------------------------------------
# 4. Eigenvalues and Eigenvectors
# ----------------------------------------
# Using covariance matrix for eigen calculation
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigen_values)
print("\nEigenvectors:\n", eigen_vectors)

# ----------------------------------------
# 5. Simple Linear Regression (for comparison)
# ----------------------------------------
# Formula: y = mx + c

# Calculate slope (m)
m = np.sum((X - mean_X) * (Y - mean_Y)) / np.sum((X - mean_X)**2)

# Calculate intercept (c)
c = mean_Y - m * mean_X

print("\nLinear Regression Equation: y = {:.2f}x + {:.2f}".format(m, c))

# Predict values
Y_pred = m * X + c

print("\nPredicted Values:", Y_pred)
