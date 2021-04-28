import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linear_classification.basic_logistic_unit import sigmoid
from solving_for_optimal_weights.cross_entropy import cross_entropy

sns.set()

def gradient_descent_l2(X, y, y_hat, W, loss_function, learning_rate=0.1, steps=100, l2_penalty=0.1):
    for step in range(steps):
        if step % 10 == 0:
            print(loss_function(y, y_hat))
        W += learning_rate * ( X.T.dot(y - y_hat) - l2_penalty * W)
        y_hat = sigmoid(X.dot(W))
    
    return W

if __name__ == '__main__':
    number_of_samples = 100
    dimensions = 2
    X = np.random.randn(number_of_samples, dimensions)

    # Center the first 50 points at (-2,-2)
    X[:50,:] = X[:50,:] - 2*np.ones((50,dimensions))

    # Center the second 50 points at (2,2)
    X[50:,:] = X[50:,:] + 2*np.ones((50,dimensions))

    y = np.array([0]*50 + [1]*50)

    X = np.concatenate((np.ones((number_of_samples, 1)), X), axis=1 )

    W = np.random.randn(dimensions+1)
    z = X.dot(W)

    y_hat = sigmoid(z)

    W = gradient_descent_l2(X, y, y_hat, W, loss_function=cross_entropy, learning_rate=0.1, steps=100, l2_penalty=0.1)
    print("Final W:", W)

    plt.figure()
    plt.scatter(X[:,1], X[:,2], c=y, s=100, alpha=0.5)
    x_axis = np.linspace(-6, 6, 100)
    y_axis = - (W[0] + x_axis * W[1]) / W[2]
    y_axis_real = -x_axis
    plt.plot(x_axis, y_axis, '--r', label='Predicted')
    plt.plot(x_axis, y_axis_real, '.b', label='Actual')
    plt.legend()
    plt.show()