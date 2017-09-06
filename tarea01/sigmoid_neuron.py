import numpy as np

class SigmoidNeuron:

    def __init__(self, w=np.empty(0), bias=0.5):
        self.output = None
        self.delta = None
        self.weights = w
        self.bias = bias
        self.z = None

    def set_weights(self, w):
        self.weights=w

    def set_bias(self, bias):
        self.bias = bias

    def feed(self, i):
        self.z = (np.sum(self.weights*i) + self.bias)
        self.output = 1.0/(1.0 + np.exp(-self.z))
        # self.output = 1.0 if self.output > 0.5 else 0.0
        return self.output

    def number_of_weights(self):
        return np.size(self.weights)

    def calculate_delta(self, error):
        self.delta = error * self.output* (1.0 - self.output)
        return self.delta

    def update_weights(self, inputs, learning_rate=0.05):
        self.weights = self.weights + (learning_rate*self.delta*inputs)
        self.bias += learning_rate*self.delta





