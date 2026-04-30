import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(42)

X = np.sort(np.random.rand(50)).reshape(-1,1)
y = np.sin(2*np.pi*X) + np.random.normal(0,0.2,50).reshape(-1,1)

degrees = [1,4,15]

for i,d in enumerate(degrees):
    plt.subplot(1,3,i+1)

    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X,y)

    X_test = np.linspace(0,1,100).reshape(-1,1)
    y_pred = model.predict(X_test)

    plt.scatter(X,y,color='green')
    plt.plot(X_test,y_pred,color='orange')

    if d==1: plt.title("Underfitting")
    elif d==4: plt.title("Good Fit")
    else: plt.title("Overfitting")

plt.show()