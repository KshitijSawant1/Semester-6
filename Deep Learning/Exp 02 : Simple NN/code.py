import math

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Input values (example)
weight = 70    # input 1
height = 170   # input 2

# Weights (as shown in diagram)
w1 = 0.01
w2 = 0.02
w3 = 0.03
w4 = 0.04
w5 = 0.05
w6 = 0.06

# Bias values
b1 = 0.5
b2 = 0.5
b3 = 0.5

# ----- Hidden Layer Computation -----
h1_input = (weight * w1) + (height * w2) + b1
h1_output = sigmoid(h1_input)

h2_input = (weight * w3) + (height * w4) + b2
h2_output = sigmoid(h2_input)

# ----- Output Layer Computation -----
o1_input = (h1_output * w5) + (h2_output * w6) + b3
o1_output = sigmoid(o1_input)

# Final Output
print("Hidden neuron h1 output:", h1_output)
print("Hidden neuron h2 output:", h2_output)
print("Final output (o1):", o1_output)

# Binary classification decision
if o1_output >= 0.5:
    print("Predicted Class: Class 1")
else:
    print("Predicted Class: Class 0")
