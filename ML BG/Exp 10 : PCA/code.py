# Experiment 10: Principal Component Analysis (PCA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------
# 1. Load Dataset (Sklearn Iris)
# ----------------------------------------
data = load_iris()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)

# ----------------------------------------
# 2. Standardize the Dataset
# ----------------------------------------
sc = StandardScaler()
X_std = sc.fit_transform(X)

# ----------------------------------------
# 3. Compute Covariance Matrix
# ----------------------------------------
cov_matrix = np.cov(X_std.T)
print("\nCovariance Matrix:\n", cov_matrix)

# ----------------------------------------
# 4. Compute Eigenvalues and Eigenvectors
# ----------------------------------------
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigen_values)
print("\nEigenvectors:\n", eigen_vectors)

# ----------------------------------------
# 5. Sort Eigenvalues and Eigenvectors
# ----------------------------------------
sorted_index = np.argsort(eigen_values)[::-1]

eigen_values = eigen_values[sorted_index]
eigen_vectors = eigen_vectors[:, sorted_index]

# Select top k components (k = 2)
k = 2
principal_components = eigen_vectors[:, :k]

# ----------------------------------------
# 6. Transform Data (Projection)
# ----------------------------------------
X_pca_manual = np.dot(X_std, principal_components)

# ----------------------------------------
# 7. PCA using sklearn (Verification)
# ----------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# ----------------------------------------
# 8. Visualization (2D Projection)
# ----------------------------------------
plt.figure(figsize=(8,6))

for i in np.unique(y):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=f"Class {i}"
    )

plt.title("PCA - 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# ----------------------------------------
# 9. Reconstruction (Inverse Transform)
# ----------------------------------------
X_reconstructed = np.dot(X_pca_manual, principal_components.T)

print("\nReconstructed Data (first 5 rows):\n", X_reconstructed[:5])