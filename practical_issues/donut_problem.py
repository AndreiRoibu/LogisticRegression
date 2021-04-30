import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import seaborn as sns
sns.set()

from linear_classification.basic_logistic_unit import sigmoid
from e_commerce_example.logistic_training import cross_entropy


def generate_donut_data(number_of_points, dimensions, inner_radius, outer_radius):
    # The distance from origin = R + random normal
    # The theta angle =  Uniformly distributed from (0, 2*pi)

    R1 = np.random.randn(number_of_points//2) + inner_radius
    R2 = np.random.randn(number_of_points//2) + outer_radius

    theta1 = 2 * np.pi * np.random.random(number_of_points//2)
    theta2 = 2 * np.pi * np.random.random(number_of_points//2)

    X_inner = np.concatenate( [ [R1 * np.cos(theta1)], [R1 * np.sin(theta1)] ] ).T
    X_outer = np.concatenate( [ [R2 * np.cos(theta2)], [R2 * np.sin(theta2)] ] ).T

    X = np.concatenate([ X_inner, X_outer ])
    y = np.array([0]*(number_of_points//2) + [1]*(number_of_points//2))

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

    return X, y 

def gradient_descent_l2_donut_problem(X, y, learning_rate=0.0001, epochs=5000, l2_norm=0.1):
    w = np.random.randn(X.shape[1])
    z = X.dot(w)
    y_hat = sigmoid(z)

    errors = []
    for epoch in range(epochs):
        error = cross_entropy(y, y_hat)
        errors.append(error)
        if epoch % 100 == 0:
            print("Epoch {}:{}".format(epoch, error))
        w += learning_rate * ( X.T.dot(y - y_hat) - l2_norm * w )
        
        y_hat = sigmoid(X.dot(w))

    plt.figure()
    plt.plot(errors)
    plt.title("Cross-entropy Loss/Iteration")
    plt.show()

    print("Final w:", w)
    print("Final classification rate:", 1 - np.abs(y - np.round(y_hat)).sum() / X.shape[0])

    return w

if __name__ == '__main__':
    number_of_points = 1000
    dimensions = 2
    inner_radius = 5
    outer_radius = 10
    learning_rate=0.0001
    epochs=5000
    l2_norm = 0.1
    
    X, y = generate_donut_data(number_of_points, dimensions, inner_radius, outer_radius)

    # Generate the ones + an additional column of r = sqrt(x^2 + y^2) 
    # This is the radious of a point, and makes data points linearly separable

    ones = np.ones((number_of_points, 1))
    radiuses = np.sqrt( (X*X).sum(axis=1) ).reshape(-1,1)
    X = np.concatenate((ones, radiuses, X), axis=1)

    _ = gradient_descent_l2_donut_problem(X, y, learning_rate, epochs, l2_norm)

    # Classification does not appear to depend on the X and y term, but on the bias and the summed radiuses