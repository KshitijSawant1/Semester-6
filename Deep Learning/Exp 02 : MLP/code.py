import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc,accuracy_score
from sklearn.preprocessing import label_binarize


# -----------------------------
# Load Wine Dataset
# -----------------------------
data = load_wine()
X = data.data
y = data.target

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Create MLP Model
# -----------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 30),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
mlp.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = mlp.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# =============================
# GRAPH 1: Loss Curve
# =============================
plt.figure()
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve of MLP (Wine Dataset)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# =============================
# Confusion Matrix in Text Form
# =============================

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Text Format):\n")
print(cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
probs = mlp.predict_proba(X_test)

plt.figure()
plt.hist(probs.max(axis=1))
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()
