from Math import *
import numpy as np
from Plot import *

# Created By Gabriel Fallik on 11/11/2025 For Deep Learning Project. Use for educational purposes.

def init_param():
    kernel = np.random.randn(3, 3) * np.sqrt(2 / 9) #Normal distribution convolution
    w = np.random.randn(10, 169) * np.sqrt(2 / 169) #Normal distribution FC layer
    b = np.zeros((10, 1))

    #initialise moments for ADAM
    m_kernel = np.zeros_like(kernel)
    v_kernel = np.zeros_like(kernel)
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = np.zeros_like(b)
    v_b = np.zeros_like(b)

    return w, b, kernel, m_kernel, v_kernel, m_w, v_w, m_b, v_b

def convolution(image, kernel, stride=1, padding=0):

    #Border intensity zero padding
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    #Height and width of image and kernel
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape

    #output dimensions
    o_h = (i_h - k_h) // stride + 1
    o_w = (i_w - k_w) // stride + 1
    output = np.zeros((o_h, o_w))

    #convolution
    for i in range (o_h):
        for j in range (o_w):
            output[i, j] = np.sum(image[i*stride:i*stride+k_h, j*stride:j*stride+k_w]*kernel)
    return output

def max_pooling(image, poolsize=2, stride=2):

    #dimensions initialization
    i_h, i_w = image.shape
    o_h = (i_h - poolsize) // stride + 1
    o_w = (i_w - poolsize) // stride + 1
    output = np.zeros((o_h, o_w))

    #Pooling (Sample maximum value in poolsize x poolsize area)
    for i in range (o_h):
        for j in range (o_w):
            output[i, j] = np.max(image[i*stride:i*stride+poolsize, j*stride:j*stride+poolsize])
    return output

def one_hot(y):
    y = int(y)
    n_classes = 10
    one_hot_y = np.zeros((n_classes, 1))  #(10, n)
    one_hot_y[y, 0] = 1  #Valeur y mis a la ligne y
    return one_hot_y #(10,1)

def forward(w_fc, b_fc, x, kernel):
    #Normalization
    #x = x / 255.0

    #Architecture: 1 Conv -> 1 pool -> 1 Fc layer
    x = x.reshape(28, 28)
    conv1 = convolution(x, kernel)
    relu1 = ReLU(conv1)
    pool1 = max_pooling(relu1)

    #Fully connected layer
    flattened = pool1.flatten().reshape(-1, 1)
    z = np.dot(w_fc, flattened) + b_fc
    a = softmax(z) #a returns after softmax activation, Sum(xi*wi + b), y(10,1) "probabilities"

    return conv1, relu1, pool1, z, a

#find gradients
def backward(conv1, relu1, pool1, a_fc, w_fc, y, x):
    one_hot_y = one_hot(y)
    m = y.size

    #output gradient (how wrong is the prediction)
    dz_fc = a_fc - one_hot_y

    #FC gradient (how much to change)
    dw_fc = np.dot(dz_fc, pool1.flatten().reshape(1, -1)) / m
    db_fc = np.sum(dz_fc, axis=1, keepdims=True) / m

    #backprop to pooling layer (how errors affect pooled features)
    d_flat = np.dot(w_fc.T, dz_fc).flatten() #(169,)
    d_pool = d_flat.reshape(pool1.shape) #(13,13)

    #backprop to convolution (Errors in the convolution layer)
    d_conv = np.zeros_like(conv1)
    for i in range(pool1.shape[0]):
        for j in range(pool1.shape[1]):
            #assigns gradients only to max values selected in the max pooling
            if pool1[i, j] == relu1[i*2, j*2]:
                d_conv[i*2, j*2] = d_pool[i,j]

    #compute the kernel gradient
    d_kernel = convolution(x.reshape(28,28), d_conv)
    return d_kernel, dw_fc, db_fc

def ADAM(w, dw, m, v, time_step, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * dw #momentum estimate
    v = beta2 * v + (1 - beta2) * dw * dw #RMSprop-like

    # learning rate correction
    m_hat = m / (1 - beta1 ** time_step)
    v_hat = v / (1 - beta2 ** time_step)

    effective_lr = lr / (np.sqrt(v_hat) + epsilon)
    w -= effective_lr * m_hat

    return m, v, w

def update_param(kernel, w_fc, b_fc, d_kernel, dw_fc, db_fc, learning_rate):
    kernel -= learning_rate * d_kernel
    w_fc -= learning_rate * dw_fc
    b_fc -= learning_rate * db_fc
    return kernel, w_fc, b_fc

def get_prediction(a2):
    assert a2.shape[0] == 10
    return np.argmax(a2, axis=0)

def grad_descent(x_train, y_train, iterations, learning_rate, losses, accuracies):
    w_fc, b_fc, kernel, m_kernel, v_kernel, m_w, v_w, m_b, v_b = init_param() #HE/ADAM initialisation

    n_data = x_train.shape[0]
    batch_size = 32

    beta1 = 0.9 #momentum decay
    beta2 = 0.999 #RMSDrop decay
    epsilon = 1e-8 #prevents /0

    for i in range(iterations):
        print(f"🔄 ITERATION {i+1}/{iterations}")

        batch_index = np.random.choice(n_data, batch_size, replace=False)
        x = x_train[batch_index]
        y = y_train[batch_index]

        # initialize gradients to 0
        d_kernel_sum = np.zeros_like(kernel)
        dw_fc_sum = np.zeros_like(w_fc)
        db_fc_sum = np.zeros_like(b_fc)
        total_loss = 0
        correct_predictions = 0

        for x, y in zip(x, y):
            conv1, relu1, pool1, z_fc, a_fc = forward(w_fc, b_fc, x, kernel)
            d_kernel, dw_fc, db_fc = backward(conv1, relu1, pool1, a_fc, w_fc, y, x)

            #accumulate gradients
            d_kernel_sum += d_kernel
            dw_fc_sum += dw_fc
            db_fc_sum += db_fc

            #compute loss
            one_hot_y = one_hot(y)  # Convert labels to one-hot
            loss = -np.sum(one_hot_y * np.log(a_fc + 1e-8))  # Cross-entropy loss
            total_loss += loss

            #compute accuracy
            correct_predictions += (get_prediction(a_fc) == y)

        #Average gradients over a batch
        d_kernel_sum /= batch_size
        dw_fc_sum /= batch_size
        db_fc_sum /= batch_size

        #Adaptive moment learning
        m_kernel, v_kernel, kernel = ADAM(kernel, d_kernel_sum, m_kernel, v_kernel, i+1, learning_rate,
                                          beta1, beta2, epsilon)
        m_w, v_w, w_fc = ADAM(w_fc, dw_fc_sum, m_w, v_w, i+1, learning_rate, beta1, beta2, epsilon)
        m_b, v_b, b_fc = ADAM(b_fc, db_fc_sum, m_b, v_b, i+1, learning_rate, beta1, beta2, epsilon)

        #update parameters
        kernel, w_fc, b_fc = update_param(kernel, w_fc, b_fc, d_kernel_sum, dw_fc_sum, db_fc_sum, learning_rate)

        #compute batch loss
        avg_loss = total_loss / batch_size
        avg_accuracy = (correct_predictions / batch_size) * 100
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)
        print(f"📉 Avg Loss: {avg_loss:.6f}")
        print(f"Accuracy: {avg_accuracy}")

        if (i % 25) == 0:
            plot(losses, accuracies)

    return kernel, w_fc, b_fc, losses