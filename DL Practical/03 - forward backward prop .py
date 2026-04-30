import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivative(x): return x*(1-x)

# Random dataset
np.random.seed(0)
X = np.random.rand(5,2)      # 5 samples, 2 features
y = np.random.randint(0,2,(5,1))  # binary output (0 or 1)

# Forward Only
W1 = np.random.rand(2,1); b1 = 0; loss_f = []
for _ in range(100):
    out = sigmoid(np.dot(X,W1)+b1)
    loss_f.append(np.mean((y-out)**2))

# Forward + Backprop
W2 = np.random.rand(2,1); b2 = 0; lr = 0.1; loss_b = []
for _ in range(100):
    out = sigmoid(np.dot(X,W2)+b2)
    loss_b.append(np.mean((y-out)**2))
    d = (y-out)*sigmoid_derivative(out)
    W2 += np.dot(X.T,d)*lr
    b2 += np.sum(d)*lr

print("Input X:\n", X)
print("Target y:\n", y)
print("Final Output:\n", out)

plt.plot(loss_f,label="Forward Only")
plt.plot(loss_b,label="With Backprop")
plt.xlabel("Iterations"); plt.ylabel("Loss")
plt.title("Loss Comparison"); plt.legend()
plt.show()