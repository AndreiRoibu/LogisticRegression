import numpy as np
from e_commerce_example.e_commerce_preprocessing import get_binary_data
from linear_classification.basic_logistic_unit import sigmoid

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(y, y_hat):
    return np.mean(y == y_hat)

if __name__ == '__main__':
    X_train, y_train, _, _ = get_binary_data()
    W = np.random.randn(X_train.shape[1])
    b = 0

    y_hat = forward(X_train,W,b) # P of y given X
    y_hat = np.round(y_hat)

    print("Score:", classification_rate(y_train, y_hat))

    scores = 0

    for _ in range(10000):
        W = np.random.randn(X_train.shape[1])
        b = 0
        y_hat = forward(X_train,W,b)
        y_hat = np.round(y_hat)
        score = classification_rate(y_train, y_hat)
        scores += score

    print("Mean Score after 10,000 repetitions:", scores/10000)