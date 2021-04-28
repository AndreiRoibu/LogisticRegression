import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from e_commerce_example.e_commerce_preprocessing import get_binary_data
from e_commerce_example.logistic_prediction import forward, classification_rate

sns.set()

def cross_entropy(y, y_hat):
    return -np.mean( y * np.log(y_hat) + (1-y) * np.log(1-y_hat) )

def logistic_training(X_train, y_train, X_test, y_test, W, b, learning_rate=0.001, iterations=10000):
    train_costs = []
    test_costs = []
    learning_rate = 0.001
    iterations = 10000

    for i in range(iterations):
        y_hat_train = forward(X_train, W, b)
        y_hat_test = forward(X_test, W, b)

        cost_train = cross_entropy(y_train, y_hat_train)
        cost_test = cross_entropy(y_test, y_hat_test)
        
        train_costs.append(cost_train)
        test_costs.append(cost_test)

        W -= learning_rate * X_train.T.dot(y_hat_train - y_train)
        b -= learning_rate * (y_hat_train - y_train).sum()

        if i % 1000 == 0:
            print (i, cost_train, cost_test)
        
    print("Final train classification_rate:", classification_rate(y_train, np.round(y_hat_train)))
    print("Final test classification_rate:", classification_rate(y_test, np.round(y_hat_test)))

    plt.figure()
    plt.plot(train_costs, label='train cost')
    plt.plot(test_costs, label='test cost')
    plt.legend()
    plt.show()

    return W, b

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_binary_data()
    W = np.random.randn(X_train.shape[1])
    b = 0

    learning_rate = 0.001
    iterations = 10000

    _, _ = logistic_training(X_train, y_train, X_test, y_test, W, b, learning_rate, iterations)