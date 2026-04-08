# Experiment 03: Logistic Regression using sklearn dataset

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------
# 1. Load Dataset (Sklearn Built-in)
# ----------------------------------------
data = load_breast_cancer()

X = data.data          # Features
y = data.target        # Target (0 or 1)

print("Dataset Loaded")
print("Feature Shape:", X.shape)
print("Target Classes:", np.unique(y))

# ----------------------------------------
# 2. Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# 3. Model Training
# ----------------------------------------
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# ----------------------------------------
# 4. Prediction
# ----------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------------------
# 5. Evaluation
# ----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# ----------------------------------------
# 6. Display Sample Predictions
# ----------------------------------------
print("\nSample Predictions:")
print("Actual  Predicted  Probability")

for i in range(10):
    print(f"{y_test[i]}        {y_pred[i]}        {y_prob[i]:.4f}")