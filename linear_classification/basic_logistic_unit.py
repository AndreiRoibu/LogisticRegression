import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z) )

if __name__ == '__main__':

    number_of_data_points = 100
    data_dimensionality = 2

    X = np.random.randn(number_of_data_points, data_dimensionality)
    bias = np.ones((number_of_data_points, 1))

    X = np.concatenate((bias, X), axis=1)

    weights = np.random.randn(data_dimensionality + 1)

    z = X.dot(weights)

    y_hat = sigmoid(z)

    print(y_hat)