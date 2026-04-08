# Experiment 08: Backpropagation and Multi-Layer Perceptron

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------
# 1. Load Dataset (Sklearn Built-in)
# ----------------------------------------
data = load_digits()

X = data.data       # Features
y = data.target     # Target (0–9 classification)

print("Dataset Loaded")
print("Shape:", X.shape)

# ----------------------------------------
# 2. Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# 3. Create MLP Model
# ----------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(100,),   # One hidden layer with 100 neurons
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# ----------------------------------------
# 4. Train Model (Backpropagation happens internally)
# ----------------------------------------
model.fit(X_train, y_train)

# ----------------------------------------
# 5. Prediction
# ----------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------
# 6. Evaluation
# ----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

# ----------------------------------------
# 7. Sample Predictions
# ----------------------------------------
print("\nActual vs Predicted:")
for i in range(10):
    print(f"{y_test[i]} -> {y_pred[i]}")