import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import seaborn as sns
sns.set()
from practical_issues.donut_problem import gradient_descent_l2_donut_problem as gradient_descent_l2_XOR_problem

if __name__ == '__main__':
    number_of_points = 4
    dimensions = 2

    #XOR data:
    X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    ])
    y = np.array([0, 1, 1, 0])

    learning_rate=0.01
    epochs=10000
    l2_norm = 0.01
    
    # Generate the ones + an additional colum transforming the problem into 3D from 2D
    # This way, we get a plane, which makes the data linearly separable

    ones = np.ones((number_of_points, 1))
    plane = (X[:,0] * X[:,1]).reshape(X.shape[0],1)
    X = np.concatenate((ones, plane, X), axis=1)

    _ = gradient_descent_l2_XOR_problem(X, y, learning_rate, epochs, l2_norm)

    # Classification does not appear to depend on the X and y term, but on the bias and the summed radiuses