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
        

if __name__ == '__main__':
    get_data()