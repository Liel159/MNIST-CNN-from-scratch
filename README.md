# MNIST-CNN-from-Scratch

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)

This repository illustrates how to build and train a Convolutional Neural Netwok
from scratch using only NumPy and built-in Math functions. It covers the entire process of
the training loop, including convolution, pooling, forward propagation, backpropagation, loss calculation, etc. 

<img src="number3.jpg" alt="number 3" width="50%"><br>

You may also need to find more thorough explanations on the internet to fully understand each part of the code. Here, only comments
about the title of each step are provided. 

This project is a part of the learning journey in deep learning based mainly on the works of
Ian Goodfellow, Yoshua Bengio and Aaron Courville "*Deep Learning*" Book.

The model is trained and evaluated on the MNIST dataset, which consists of handwritten digit images with its respective
label 'y' representing the true value.

## Results

After training, the model achieves around 90% accuracy on the test set, the loss decreases and accuracy increases over epochs as shown below:

<img src="Training_Progress.png" alt="Training Progress Graph" width="100%"><br>

## Usage

Please install the required libraries before running and exploring the code: `numpy` and `matplotlib`.

To run please do the following two commands in your terminal:

```bash
python cnn_trainer.py
python predictions.py
```

## Sources

This poject is inspired mainly on the work "*Deep Learning*" by Ian Goodfellow, Yoshua Bengio and Aaron Courville.