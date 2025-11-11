import pickle
import cv2
from CNN import *
import numpy as np
from src.CNN import convolution, max_pooling
import os

def save_model(kernel, w_fc, b_fc, filename):
    with open(filename, 'wb') as f:
        pickle.dump({"kernel":kernel, "w_fc":w_fc, "b_fc": b_fc}, f)
    print(f"Model saved successfully as {filename}")

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model["kernel"], model["w_fc"], model["b_fc"]

def predict(x, kernel, w_fc, b_fc):
    conv1 = convolution(x, kernel)
    relu1 = ReLU(conv1)
    pool1 = max_pooling(relu1)
    flattened = pool1.flatten().reshape(-1, 1)
    z = np.dot(w_fc, flattened) + b_fc
    a = softmax(z)
    return get_prediction(a)

def load_image(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: File '{filename}' not found!")

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Error: OpenCV failed to load '{filename}'. Check file format.")

    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(28, 28)
    return img