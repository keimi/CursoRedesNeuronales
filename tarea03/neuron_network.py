from neuron_layer import NeuronLayer
from neuron_layer import InputError
import numpy as np

class ArchError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super(ArchError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors


class NeuronNetwork:

    def __init__(self, number_of_inputs=0, learning_rate=0.05):
        self.layers = []
        self.number_of_inputs = number_of_inputs
        self.learning_rate = learning_rate
        self.original_inputs = ''

    def add_layer(self, number_of_neurons):
        if np.size(self.layers) > 0:
            self.layers.append(NeuronLayer(number_neurons=number_of_neurons, input_size=self.layers[-1].number_of_neurons()))
        else:
            self.layers.append(NeuronLayer(number_neurons=number_of_neurons, input_size=self.number_of_inputs))

    def number_of_layers(self):
        return len(self.layers)

    def feed(self, inputs):
        self.original_inputs = inputs
        if np.size(inputs) != self.number_of_inputs:
            raise InputError("Wrong number of inputs")
        else:
            output = inputs
            for layer in self.layers:
                output = layer.feed(output)
            return output

    def backpropagate_error(self, expected):

        if expected.__class__ == 0.0.__class__ or expected.__class__ == (0).__class__ :
            expected = np.array([expected])
        elif expected.__class__ == [].__class__:
            expected = np.array(expected)


        if np.size(self.layers) > 0:
            next_layer = self.layers[-1]
            for indn, neuron in enumerate(next_layer.neurons):
                error = expected[indn] - neuron.output
                neuron.calculate_delta(error)

            n_layers =  len(self.layers)
            for layer in reversed(self.layers[0:n_layers-1]):
                layer.backpropagate_error(next_layer)
                next_layer = layer

    def update_weights(self):
        inputs = self.original_inputs
        for layer in self.layers:
            layer.update_weights(inputs=inputs, learning_rate=self.learning_rate)
            inputs = np.empty(0)
            for neuron in layer.neurons:
                inputs  = np.append(inputs, neuron.output)

    def __str__(self):
        s = 'neuron: ' + str(self.original_inputs) +  '\n'
        for layer in self.layers:
            s = s + '{ \n'
            for neuron in layer.neurons:
                s = s + '\t{ '
                for w in neuron.weights:
                    s = s + str(w) + ' '
                s = s +'} b: ' + str(neuron.bias) + ' z: ' + str(neuron.z) +  ' o: ' + str(neuron.output) +  ' d:' + str(neuron.delta)  + '\n'
            s =  s + '}\n'
        return s

    def to_list(self):
        ret = []
        for layer in self.layers:
            for neuron in layer.neurons:
                vec = np.hstack((np.array(neuron.weights), neuron.bias))
                ret.append(vec)
                # if not ret.any():
                #     ret = vec
                # else:
                #     ret =  np.vstack((ret, vec))

        return ret

    def from_list(self, array):
        prev_neurons = 0
        for index_layer, layer in enumerate(self.layers):
            for index_neuron, neuron in enumerate(layer.neurons):
                neuron.from_nparray(array[prev_neurons])
                prev_neurons+=1












