import numpy as np
import pandas as pd
import os

# This allows scripts from other folders to import this file
directory_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
directory_path = directory_path.replace('LogisticRegression/utils', '')

def get_data():
    df = pd.read_csv(directory_path + 'large_files/ecommerce_data.csv')
    data = df.values # We extract the numerical values
    np.random.shuffle(data) # We shuffle the data

    # We split the data into features and labels
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int32)

    # We one-hot encode the data
    samples, features = X.shape
    X_new = np.zeros((samples, features + 3))
    X_new[:, 0:(features-1)] = X[:, 0:(features-1)] # Non-categorial data

    for sample in range(samples): 
        time_of_day = int(X[sample, features-1])
        X_new[sample, time_of_day + features - 1] = 1  # Categorial data

    X = X_new
    del X_new
    
    # We split and normalize

    X_train = X[:-100]
    y_train = y[:-100]
    X_test = X[-100:]
    y_test = y[-100:]

    for i in (1, 2):
        mean = X_train[:,i].mean()
        std = X_train[:,i].std()
        X_train[:,i] = (X_train[:, i] - mean) / std
        X_test[:,i] = (X_test[:, i] - mean) / std

    return X_train, y_train, X_test, y_test

def get_binary_data():
    # Only returns the data from the first 2 classes
    X_train, y_train, X_test, y_test = get_data()
    return X_train[y_train <=1 ], y_train[y_train <=1 ], X_test[y_test <=1 ], y_test[y_test <=1 ]

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data()
    print(y_test)
    X_train, y_train, X_test, y_test = get_binary_data()
    print(y_test)