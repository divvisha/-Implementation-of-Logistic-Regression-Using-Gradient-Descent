# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.


## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: Divyashree B S
RegisterNumber:  212221040044

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data= np.loadtxt("/content/ex2data1 (2).txt", delimiter=',')
X= data[:, [0,1]]
y= data[:, 2]

print("Array value of X:")
X[:5]

print("Array value of Y:")
y[:5]

print("Exam 1-score graph:")
plt.figure()
plt.scatter(X[y==1][:, 0],X[y==1][:, 1], label="Admitted")
plt.scatter(X[y==0][:, 0],X[y==0][:, 1], label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print("Sigmoid function graph:")
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return J,grad
  
print("X_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

print("Y_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y) / X.shape[0]
  return grad 
  
print("Print res.x:")
X_train = np.hstack((np.ones((X.shape[0], 1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y), method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max= X[:,0].min()-1, X[:,0].max()+1
  y_min, y_max= X[:,0].min()-1, X[:,0].max()+1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted") 
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,y)

print("Probability value:")
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
  
print("Prediction value of mean:")
np.mean(predict(res.x,X) == y)
```

## Output:

<img width="271" alt="ex5 op1" src="https://user-images.githubusercontent.com/127508123/235950297-6bfe9108-e3e4-4028-8939-0fcab93385e9.png">

<img width="281" alt="ex5 op2" src="https://user-images.githubusercontent.com/127508123/235950350-226cdff7-240e-4020-a9c4-31e5315a7731.png">

<img width="449" alt="ex5 op3" src="https://user-images.githubusercontent.com/127508123/235950395-2952842b-0ab0-47e4-9e32-ed452f465e2e.png">

<img width="408" alt="ex5 op4" src="https://user-images.githubusercontent.com/127508123/235950453-ede6b31e-7cd8-4032-b25d-aac2189178d2.png">

<img width="406" alt="ex5 op5" src="https://user-images.githubusercontent.com/127508123/235950566-d96e228f-a6f3-4360-94c4-29c285a82f05.png">

<img width="456" alt="ex5 op6" src="https://user-images.githubusercontent.com/127508123/235950653-c68be76c-0631-42e3-ab01-f44b14edf850.png">

<img width="471" alt="ex5 op7" src="https://user-images.githubusercontent.com/127508123/235950690-5449055d-d34e-451a-8617-0f444a91a0b3.png">

<img width="501" alt="ex5 op8" src="https://user-images.githubusercontent.com/127508123/235950803-a5ba6f10-be13-4a7e-9c13-aa5c2723d8a7.png">

<img width="362" alt="ex5 op9" src="https://user-images.githubusercontent.com/127508123/235950828-6ebcbf18-4a98-4dba-8d55-72d7f3746756.png">

<img width="277" alt="ex5 op10" src="https://user-images.githubusercontent.com/127508123/235950855-54daf225-5d5b-4fc6-a52a-1771f814bd59.png">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

