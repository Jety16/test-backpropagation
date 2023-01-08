import numpy as np

# loss function
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# loss function derivative
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# activation function (hiperbolic tangent)
def tanh(x):
    return np.tanh(x)

# activation function derivative
def tanh_prime(x):
    return 1 - np.tanh(x)**2
