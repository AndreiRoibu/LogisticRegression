import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.utils import shuffle
import argparse

from facial_expression_recognition.utils import get_binary_data, get_data, sigmoid, sigmoid_cost, error_rate, y_one_hot_indication, softmax

class LogisticModel_Binary(object):
    def __init__(self):
        pass

    def fit(self, X, y, learning_rate=1e-6, regularisation=0.0, epochs=120000, show_figure=False):
        X, y = shuffle(X, y)
        X_train, y_train = X[:-1000], y[:-1000]
        X_validation, y_validation = X[-1000:], y[-1000:]
        del X,y

        number_subjects, dimensions = X_train.shape
        self.W = np.random.randn(dimensions) / np.sqrt(dimensions)
        self.b = 0

        costs = []
        best_validation_error = 1

        for epoch in range(epochs):
            # Forward propagation + cost calculation; Probability of y given X
            y_hat = self.forward(X_train)

            # Gradient Descent
            self.W -= learning_rate * (X_train.T.dot(y_hat - y_train) + regularisation * self.W)
            self.b -= learning_rate * ((y_hat - y_train).sum() + regularisation * self.b)

            if epoch % 20 == 0:
                y_hat_validation = self.forward(X_validation)
                cost_validation = sigmoid_cost(y_validation, y_hat_validation)
                costs.append(cost_validation)
                error = error_rate(y_validation, np.round(y_hat_validation))
                print("Epcoh: {} | Cost: {} | Error: {}".format(epoch, cost_validation, error))

                if error < best_validation_error:
                    best_validation_error = error
        
        print("Best validation error:", best_validation_error)

        if show_figure:
            plt.figure()
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        y_hat = self.forward(X)
        return np.round(y_hat)

    def score(self, X, y):
        prediction = self.predict(X)
        return 1 - error_rate(y, prediction)

class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self, X, y, X_validation, y_validation, learning_rate=1e-7, regularisation=0.0, epochs=10000, show_figure=False):
        
        y_dimensions = len(set(y))
        y_one_hot = y_one_hot_indication(y)
        y_one_hot_validation = y_one_hot_indication(y_validation)
        
        _, dimensions = X.shape
        self.W = np.random.randn(dimensions, y_dimensions) / np.sqrt(dimensions)
        self.b = np.zeros(y_dimensions)

        costs = []
        best_validation_error = 1

        for epoch in range(epochs):
            # Forward propagation + cost calculation; Probability of y given X
            y_hat = self.forward(X)

            # Gradient Descent
            self.W -= learning_rate * (X.T.dot(y_hat - y_one_hot) + regularisation * self.W)
            self.b -= learning_rate * ((y_hat - y_one_hot).sum() + regularisation * self.b)

            if epoch % 20 == 0:
                y_hat_validation = self.forward(X_validation)
                cost_validation = sigmoid_cost(y_one_hot_validation, y_hat_validation)
                costs.append(cost_validation)
                error = error_rate(y_one_hot_validation, np.argmax(y_hat_validation, axis=1))
                print("Epcoh: {} | Cost: {} | Error: {}".format(epoch, cost_validation, error))

                if error < best_validation_error:
                    best_validation_error = error
        
        print("Best validation error:", best_validation_error)

        if show_figure:
            plt.figure()
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)

    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)

    def score(self, X, y):
        prediction = self.predict(X)
        return 1 - error_rate(y, prediction)

def main():

    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are binary or full')
    arguments = parser.parse_args()
    
    if arguments.mode == "binary":
        X, y = get_binary_data()
        model = LogisticModel_Binary()
        model.fit(X, y, show_fig=True)
        print("SCORE:", model.score(X, y))
    elif arguments.mode == "full":
        X_train, y_train, X_validation, y_validation = get_data()
        model = LogisticModel()
        model.fit(X_train, y_train, X_validation, y_validation, show_fig=True)
        print("SCORE:", model.score(X_validation, y_validation))
    else:
        print("ERROR! Mode NOT VALID!")

if __name__ == '__main__':
    main()
