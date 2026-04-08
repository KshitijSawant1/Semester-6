# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ----------------------------------------
# Step 2: Import Dataset (Iris)
# ----------------------------------------
data = load_iris()

# Step 3: Split into X and y
X = data.data
y = data.target

print("Dataset Shape:", X.shape)

# ----------------------------------------
# Step 4: Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# Step 5 & 6: Standardization (Feature Scaling)
# ----------------------------------------
sc = StandardScaler()

X_train = sc.fit_transform(X_train)   # Fit and transform training data
X_test = sc.transform(X_test)         # Only transform test data

# ----------------------------------------
# Step 7: Apply PCA
# ----------------------------------------
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ----------------------------------------
# Step 8: Visualization (2D Projection)
# ----------------------------------------
plt.figure(figsize=(8,6))

for i in np.unique(y_train):
    plt.scatter(
        X_train_pca[y_train == i, 0],
        X_train_pca[y_train == i, 1],
        label=f"Class {i}"
    )

plt.title("PCA - 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# ----------------------------------------
# Step 9: Mapping already done using transform()
# ----------------------------------------

# ----------------------------------------
# Step 10: Apply Logistic Regression
# ----------------------------------------
model = LogisticRegression()
model.fit(X_train_pca, y_train)

# ----------------------------------------
# Step 11: Predict Test Results
# ----------------------------------------
y_pred = model.predict(X_test_pca)

print("\nPredicted Values:", y_pred)
print("Actual Values   :", y_test)