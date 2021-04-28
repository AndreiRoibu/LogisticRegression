from os import W_OK
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linear_classification.basic_logistic_unit import sigmoid
from e_commerce_example.logistic_training import cross_entropy

sns.set()

def gradient_descent_l1(X, y, learning_rate=0.001, steps=5000, l1_penalty=3):
    
    costs = []
    W = np.random.randn(X.shape[1]) / np.sqrt(X.shape[1])
    
    for _ in range(steps):

        y_hat = sigmoid(X.dot(W))
        W -= learning_rate * ( X.T.dot(y_hat - y) + l1_penalty * np.sign(W))

        cost = cross_entropy(y, y_hat) + l1_penalty * np.abs(W).mean()
        costs.append(cost)

    plt.figure()
    plt.plot(costs)
    plt.title("Costs")
    plt.show()
    
    return W

if __name__ == '__main__':
    number_of_samples = 50
    dimensions = 50

    # Uniformly distribute numbers between -5 and 5
    X = (np.random.random((number_of_samples, dimensions)) - 0.5) * 10

    W_true = np.array([1, 0.5, -0.5] + [0] * (dimensions-3))

    y = sigmoid(X.dot(W_true))
    y = np.round(y + np.random.rand(number_of_samples) * 0.5) # Add noise with variance 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
    plt.show()

    W = gradient_descent_l1(X, y, learning_rate=0.001, steps=5000, l1_penalty=3.0)
    print("Final W:", W)

    plt.figure()
    plt.plot(W_true, label='true w')
    plt.plot(W, label='w_map')
    plt.legend()
    plt.show()