import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os

def get_data(number_validation=1000, balance_class_one = True):
    X = []
    y = []
    first = True

    # This allows scripts from other folders to import this file
    directory_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    directory_path = directory_path.replace('LogisticRegression/facial_expression_recognition', '') 
    data_path = directory_path + 'large_files/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013'
    file_path = data_path + '/fer2013.csv'

    for line in open(file_path):
        if first:
            first=False # Skips the first line
        else:
            row = line.split(',')
            y.append(int(row[0]))
            X.append([int(pixel) for pixel in row[1].split()])
    
    X = np.array(X) / 255.0
    y = np.array(y)

    X, y = shuffle(X, y)
    X_train, y_train = X[:-number_validation], y[:-number_validation]
    X_validation, y_validation = X[-number_validation:], y[-number_validation:]

    # As the classes are unbalanced, we lenghten class one by repeating it 9 times
    if balance_class_one == True:
        X0, y0 = X_train[y_train!=1, :], y_train[y_train != 1]
        X1 = X_train[y_train==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X_train = np.vstack([X0, X1])
        y_train = np.concatenate((y0, [1]*len(X1)))

    return X_train, y_train, X_validation, y_validation

def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

def sigmoid_cost(y, y_hat):
    return -(y * np.log(y_hat) + (1-y)*np.log(1-y_hat)).sum()

def error_rate(y, y_hat):
    return np.mean(y != y_hat)

def get_binary_data(balance_class_one = True):
    X = []
    y = []
    first = True

    # This allows scripts from other folders to import this file
    directory_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    directory_path = directory_path.replace('LogisticRegression/facial_expression_recognition', '') 
    data_path = directory_path + 'large_files/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013'
    file_path = data_path + '/fer2013.csv'

    for line in open(file_path):
        if first:
            first=False # Skips the first line
        else:
            row = line.split(',')
            int(row[0])
            if int(row[0]) == 0 or int(row[0]) == 1:
                y.append(int(row[0]))
                X.append([int(pixel) for pixel in row[1].split()])
    
    X = np.array(X) / 255.0
    y = np.array(y)

    X, y = shuffle(X, y)

    if balance_class_one == True:
        X0, y0 = X[y!=1, :], y[y != 1]
        X1 = X[y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        y = np.concatenate((y0, [1]*len(X1)))

    return X, y

def y_one_hot_indication(y):
    number_subjects, dimensions = len(y), len(set(y))
    one_hot_encoding = np.zeros((number_subjects, dimensions))
    for i in range(number_subjects):
        one_hot_encoding[i, y[i]] = 1
    return one_hot_encoding

def softmax(x):
    expX = np.exp(x)
    return expX / expX.sum(axis=1, keedims=True)

if __name__ == '__main__':
    get_data()