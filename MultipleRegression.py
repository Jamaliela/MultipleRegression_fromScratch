######################################################################
# Author: Elaheh Jamali
# Username: Jamalie

# Programming Assignment 1: Regression Model
#
# Purpose: In this assignment, We will use the gradient descent algorithm
# discussed in class to solve for the coefficients.  The regression model
# is able to accept an arbitrary number of input variables.
#
# Acknowledgement: Emely Alfaro Zavala for helping me to complete
# this assignment. Different articles were read for this.
#
######################################################################
import numpy as np  # library supporting large, multi-dimensional arrays and matrices.
import pandas as pd  # library to take data and creates a Python object with rows and columns
import matplotlib.pyplot as plot  # library for embedding plots
from mpl_toolkits.mplot3d import Axes3D # library for 3D model

data = pd.read_csv('FuelConsumptionCo2.csv')
print(data.shape)
print(data.head())

MC = data['MC'].values
Bid = data['Bid'].values
MarketPrice = data['Market Price'].values

# Plotting the scores as scatter plot
figure = plot.figure()
axes = Axes3D(figure)
axes.scatter(MC, Bid, MarketPrice, color='#ef1234')
plot.show()

# generating our X, Y and B
m = len(MC)
x0 = np.ones(m)
X = np.array([x0, MC, Bid]).T
# Initial Coefficients
B = np.array([0, 0, 0])
Y = np.array(MarketPrice)
alpha = 0.0001


# defining the cost function
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J


initialCost = cost_function(X, Y, B)
print("This is the initial cost:", initialCost)


# reducing our cost using Gradient Descent
def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost

    return B, cost_history


# 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

# New Values of B
print("this is the new value of B", newB)

# Final Cost of new B
print("This is the final value of B", cost_history[-1])


# RMSE (Root Mean Square Error
def RMSE(Y, Y_prediction):
    rmse = np.sqrt(sum((Y - Y_prediction) ** 2) / len(Y))
    return rmse


# Coefficient of determination
def r2_score(Y, Y_prediction):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_prediction) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


Y_prediction = X.dot(newB)

print("This is Root Mean Square Error:", RMSE(Y, Y_prediction))
print("This is the coefficient of determination:", r2_score(Y, Y_prediction))
