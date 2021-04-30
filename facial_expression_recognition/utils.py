import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os

def get_data(number_validation=1000):
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

    return X_train, y_train, X_validation, y_validation
        

if __name__ == '__main__':
    get_data()