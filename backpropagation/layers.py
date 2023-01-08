import numpy as np


class LayerBase:
    ''' Base Class '''

    def __init__(self):
        self.input = None
        self.output = None

    # Computes Y output of a layer for a given input.
    def forward_propagation(self):
        raise NotImplementedError

    # Computes dE/dX for a given dE/dY
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(LayerBase):
    def __init__(self, input_size, output_size):
        # Weight and Cost of the nodes.
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # forward_propagations it's simple
    # Just take a matrices X then calculate the dot product between 
    # X and Weigths plus the Bias (cost)
    # Y = XW + B
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    # This is the same as foward but backwards XD
    # Now we use the derivatives of the same values. but remember we are using a
    # backward_propagation. So our inputs are our outputs plus we have a learning_rate
    # What is that? Simple. remember when we use gradiant descent?
    # So yeh the learning_rate basically are the size of the "steps" we make in the gradiant
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

# this is just the function to apply
class ActivationLayer(LayerBase):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

