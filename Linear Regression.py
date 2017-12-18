#Linear Regression Algorithm Implementation from scratch in Python

import numpy as np
import matplotlib.pyplot as plt
'''
#Load Training and Testing Data Set
train = np.genfromtxt('housing.txt')
test = np.genfromtxt('housing_Predict.txt')

#Training Data for Linear Regression
X_train = train[0:,0:13]
X_train = np.c_[np.ones(X_train.shape[0]),X_train]
Y_train = train[0:,13:14]
trans = np.transpose(X_train)

#Testing Data for Linear Regression
X_test = test[0:,0:13]
X_test = np.c_[np.ones(X_test.shape[0]),X_test]
Y_test = test[0:,13:14]

'''
#Testing Linear Regression
X_train=np.array([2,3,4,5,6,7,8,9,10,11,12,13]).reshape(4,3)
Y_train=np.array([7,8,4,7])
trans = X_train.T

#Parameters Setup
rate = .001
lamda = 10
iterations = 2000
theta = np.ones((X_train.shape[1],1))
gradient = np.zeros((X_train.shape[1],1))
lossgraph = []
#Function to minimize Theta
def minimize(th,gradient):
    th = th * (1 - (rate * lamda)/384) - (rate*gradient)
    return th

#Iterations for the Gradient Descent
for i in range(iterations):
     #Hypothesis Function : h(x) = X_train.dot(theta)
     predict = X_train.dot(theta)
     error = predict - Y_train
     regularization = theta.sum()**2
     loss = ((error**2) + (lamda * regularization))
     print(loss.sum()/2 * X_train.shape[0])
     lossgraph.append(loss.sum())
     gradient = trans.dot(error)
     theta = minimize(theta,gradient)

plt.plot(np.arange(iterations),lossgraph)
plt.show()
