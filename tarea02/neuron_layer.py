from sigmoid_neuron import SigmoidNeuron
import numpy as np

class InputError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super(InputError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

class NeuronLayer:

    def __init__(self, number_neurons=0, input_size=0):
        self.neurons = []
        for i in range(number_neurons):
            self.neurons.append(SigmoidNeuron(w=np.random.rand(input_size)))

    def set_neurons(self, number_neurons=0, input_size=0):
        for i in range(number_neurons):
            self.neurons.append(SigmoidNeuron(w=np.random.rand(input_size)))

    def feed(self, inputs):
        if np.size(self.neurons) < 1:
            raise InputError("Can not feed layer without neurons")
        else:
            # Change number of neurons weights
            if np.size(inputs) != self.number_of_weights():
                for neuron in self.neurons:
                    neuron.set_weights(np.random.rand(np.size(inputs)))

        output = np.empty(0)
        for neuron in self.neurons:
            output = np.append(output, neuron.feed(inputs))
        return output

    def number_of_neurons(self):
        return np.size(self.neurons)

    def number_of_weights(self):
        if self.number_of_neurons() < 1:
            return -1
        elif self.number_of_neurons() == 1:
            return self.neurons[-1].number_of_weights()
        else:
            n = self.neurons[-1].number_of_weights()
            for neuron in self.neurons:
                if n !=neuron.number_of_weights():
                    return -1
            return n


    def backpropagate_error(self, next_layer):

        for index, neuron in enumerate(self.neurons):
            error = 0.0
            for next_neuron in next_layer.neurons:
                error += next_neuron.weights[index] * next_neuron.delta
            neuron.calculate_delta(error)

    def update_weights(self, inputs, learning_rate=0.05):
        for neuron in self.neurons:
            neuron.update_weights(inputs=inputs,learning_rate=learning_rate)








