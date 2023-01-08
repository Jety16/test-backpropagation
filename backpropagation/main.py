
import numpy as np

from network import Network
from layers import FCLayer, ActivationLayer
from utils import mse, mse_prime, tanh, tanh_prime

from keras.datasets import mnist
from keras.utils import np_utils

# XOR TEST
print('XOR TEST')

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add_layer(FCLayer(2, 3))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(3, 1))
net.add_layer(ActivationLayer(tanh, tanh_prime))

# train
net.set_loss(mse, mse_prime)
net.fit(x_train, y_train, batch=1000, learning_rate=0.1)

# predict
out = net.predict(x_train)

print('XOR TEST RESULT')
for j in range(4):
    print(f'prediction: {out[j]},  true value: {y_train[j]})')
print('\n\n\n')


# MNIST TEST
print('MNIST TEST')

# Retrieve samples from server
# 60000 samples
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype('float32')
x_train /= 255

# Encode output which is a number in range [0,9] into a vector of size 10
y_train = np_utils.to_categorical(y_train)

# Same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Create new Network
net = Network()

# input_shape=(1, 28 * 28)
# output_shape=(1, 100)
net.add_layer(FCLayer(28 * 28, 100))
net.add_layer(ActivationLayer(tanh, tanh_prime))

# input_shape=(1, 100)
# output_shape=(1, 50)
net.add_layer(FCLayer(100, 50))
net.add_layer(ActivationLayer(tanh, tanh_prime))

# input_shape=(1, 50)
# output_shape=(1, 10)
net.add_layer(FCLayer(50, 10))
net.add_layer(ActivationLayer(tanh, tanh_prime))

# training on 1000 samples
net.set_loss(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], batch=35, learning_rate=0.1)

# test on 2 samples
out = net.predict(x_test[0:10])

print('MNIST TEST RESULT')
print(f'true_value:{y_test[0:10]}, \npredict{out}')
