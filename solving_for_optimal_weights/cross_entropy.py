import numpy as np
from linear_classification.basic_logistic_unit import sigmoid

def cross_entropy(y, y_hat):
    E = 0
    for i in range(len(y)):
        if y[i] == 1:
            E -= np.log(y_hat[i])
        else:
            E -= np.log(1-y_hat[i])
    return E

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

    print("Cross Entropy:", cross_entropy(y, y_hat))
    w_closed_form = np.array([0, 4, 4])
    y_hat_closed_form = sigmoid(X.dot(w_closed_form))
    print("Cross Entropy (closed form):", cross_entropy(y, y_hat_closed_form))