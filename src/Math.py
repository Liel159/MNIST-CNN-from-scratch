import numpy as np

def ReLU(z):
    z = np.array(z)
    return np.maximum(0.01*z, z)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)  # Prevent large values before exp
    exp_z = np.exp(z)
    softmax = (exp_z / np.sum(exp_z, axis=0, keepdims=True) + 1e-8)
    return softmax

def derivative_relu(z):
    return z > 0