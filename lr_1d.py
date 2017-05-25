import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# turn X and Y to numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot the data
plt.scatter(X, Y)
plt.show()

# apply the linear regression equations to find a and b
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean()*X.sum()) / denominator
b = (Y.mean()*X.dot(X) - X.mean() * X.dot(Y)) / denominator

# calculate predicted Y
Yhat = a*X + b

# plot data with prediction line
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# calculate r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared is: ", r2)
