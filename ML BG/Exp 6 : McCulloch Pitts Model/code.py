# Experiment 06: McCulloch-Pitts Model

import numpy as np

# ----------------------------------------
# 1. Input Arrays (Example: AND Gate)
# ----------------------------------------

x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

# Target Output (AND Gate)
t = np.array([0, 0, 0, 1])

print("Input x1:", x1)
print("Input x2:", x2)
print("Target Output:", t)

# ----------------------------------------
# 2. User Input for Weights and Threshold
# ----------------------------------------

w1 = float(input("Enter W1 Weight Value: "))
w2 = float(input("Enter W2 Weight Value: "))
T  = float(input("Enter Threshold Value: "))

# ----------------------------------------
# 3. Calculate Yin
# ----------------------------------------

yin = w1 * x1 + w2 * x2

print("\nYin:", yin)

# ----------------------------------------
# 4. Initialize Output Array
# ----------------------------------------

y = np.zeros(len(yin))

# ----------------------------------------
# 5. Apply Threshold Function
# ----------------------------------------

for i in range(len(yin)):
    if yin[i] >= T:
        y[i] = 1
    else:
        y[i] = 0

# ----------------------------------------
# 6. Display Results
# ----------------------------------------

print("\nTarget Output:", t)
print("Calculated Output:", y)

# ----------------------------------------
# 7. Check Correctness
# ----------------------------------------

if np.array_equal(y, t):
    print("\nCorrect Weight And Threshold Values")
else:
    print("\nIncorrect Weights, Re-run Code")