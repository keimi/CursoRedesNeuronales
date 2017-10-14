import unittest
import numpy as np
from neuron_network import NeuronNetwork
from random import randint


def and_function(x, y):
    return x and y

def or_function(x, y):
    return x or y

def xor_function(x, y):
    return (x and not y) or (not x and y)


def create_2_2_1_network():
    network = NeuronNetwork(2, 0.5)
    network.add_layer(2)
    network.add_layer(1)
    return network

def train_n_epochs_with_function(network, f, n):
    for i in range(0, n):
        x = randint(0, 1)
        y = randint(0, 1)

        expected = f(x, y)

        inputs = np.array([x, y])
        output = network.feed(inputs)
        network.backpropagate_error(expected)
        network.update_weights()
    return network

def test_precision_with_funcction(network, f, n):
    sumTrue = 0
    errors = np.array([])
    for i in range(0, 1000):
        x = randint(0, 1)
        y = randint(0, 1)

        raw_output = network.feed(np.array([x, y]))[0]
        output = 1.0 if raw_output > 0.5 else 0.0
        expected = f(x, y)

        errors = np.append(errors, expected - raw_output)
        if output == expected:
            sumTrue += 1

    precision = sumTrue / n
    promError = np.mean(np.abs(errors))
    # print('prec: ', str(precision), 'error: ', str(promError))
    return [precision, promError]



class MyTestCase(unittest.TestCase):
    def test_and(self):
        w=np.array([1, 1])
        per= SigmoidNeuron(w,-1.5)

        i= np.array([1,1])
        self.assertGreater(per.feed(i),0.5 )
        i = np.array([1, 0])
        self.assertLess(per.feed(i), 0.5)
        i = np.array([0, 1])
        self.assertLess(per.feed(i), 0.5)
        i = np.array([0, 0])
        self.assertLess(per.feed(i), 0.5)

    def test_or(self):
        w = np.array([1, 1])
        per=SigmoidNeuron(w,-0.5)

        i = np.array([1, 1])
        self.assertGreater(per.feed(i),0.5)
        i = np.array([1, 0])
        self.assertGreater(per.feed(i), 0.5)
        i = np.array([0, 1])
        self.assertGreater(per.feed(i), 0.5)
        i = np.array([0, 0])
        self.assertLess(per.feed(i), 0.5)

    def test_and_network(self):
        net = create_2_2_1_network()
        train_n_epochs_with_function(net, and_function, 10000)
        [prec, error] = test_precision_with_funcction(net, and_function, 1000)
        print('prec: ',str(prec),'error: ', str(error))
        self.assertLess(error, 0.05)

    def test_or_network(self):
        net = create_2_2_1_network()
        train_n_epochs_with_function(net, or_function, 10000)
        [prec, error] = test_precision_with_funcction(net, or_function, 1000)
        print('prec: ',str(prec),'error: ', str(error))
        self.assertLess(error, 0.05)
        self.assertEqual(prec, 1.0)

    def test_xor_network(self):
        net = create_2_2_1_network()
        train_n_epochs_with_function(net, xor_function, 20000)
        [prec, error] = test_precision_with_funcction(net, xor_function, 1000)
        print('prec: ', str(prec), 'error: ', str(error))
        self.assertLess(error, 0.05)
        self.assertEqual(prec, 1.0)


if __name__ == '__main__':
    unittest.main()
