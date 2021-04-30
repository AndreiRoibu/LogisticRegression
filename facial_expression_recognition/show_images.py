import numpy as np
import matplotlib.pyplot as plt
from builtins import range, input

from facial_expression_recognition.utils import get_data

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    X_train, y_train, _, _ = get_data()
    while True:
        for i in range(7):
            x, y = X_train[y_train==i], y_train[y_train==i]
            N = len(y)
            j = np.random.choice(N)
            plt.imshow(x[j].reshape(48,48), cmap='gray')
            plt.title(label_map[y[j]])
            plt.show()
        prompt = input('Quit? Enter Y:\n')
        if prompt.lower().startswith('y'):
            break

if __name__ == '__main__':
    main()