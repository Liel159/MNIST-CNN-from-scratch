from sklearn.datasets import fetch_openml
import numpy as np

def import_data():
    # Importation of MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data, mnist.target.to_numpy().astype(int).reshape(-1, 1)
    data = np.hstack((y, x))  # combine 2 arrays

    m, n = data.shape
    np.random.shuffle(data)

    # Validation data set
    data_dev = data[0:1000].T
    y_dev = data_dev[0]
    x_dev = data_dev[1:n]

    # training data set
    data_train = data[1000:m].T
    y_train = data_train[0]
    y_train = y_train.reshape(-1)
    x_train = data_train[1:n]

    # Normalization
    x_train = x_train / 255.0
    x_dev = x_dev / 255.0

    return x_train, y_train, x_dev, y_dev