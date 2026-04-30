import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

weight = 70
height = 170

w1,w2,w3,w4,w5,w6 = 0.01,0.02,0.03,0.04,0.05,0.06
b1,b2,b3 = 0.5,0.5,0.5

h1 = sigmoid(weight*w1 + height*w2 + b1)
h2 = sigmoid(weight*w3 + height*w4 + b2)

o1 = sigmoid(h1*w5 + h2*w6 + b3)

print("Output:", o1)

print("Class 1" if o1 >= 0.5 else "Class 0")