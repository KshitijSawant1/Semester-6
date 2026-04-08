# Experiment 04: Support Vector Machine using sklearn

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------
# 1. Load Dataset (Sklearn Built-in)
# ----------------------------------------
data = load_breast_cancer()

X = data.data
y = data.target   # Binary classification (0,1)

print("Dataset Loaded")
print("Shape:", X.shape)

# ----------------------------------------
# 2. Split Dataset
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# 3. Create and Train Model
# ----------------------------------------
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# ----------------------------------------
# 4. Prediction
# ----------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------
# 5. Evaluation
# ----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

# ----------------------------------------
# 6. Sample Output
# ----------------------------------------
print("\nActual vs Predicted:")
for i in range(10):
    print(f"{y_test[i]} -> {y_pred[i]}")