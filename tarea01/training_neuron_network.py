from neuron_network import NeuronNetwork
import numpy as np
import matplotlib.pyplot as plt
from random import randint


# class Line:
#     m=0.0
#     n=0.0
#
#     def diag_line(self, x, y):
#
#         if self.m*x +self.n > y:
#             return 1
#         else:
#             return 0
#
#     def y(self, x):
#         return self.m*x+self.n
#
# def plot_coordinates(network):
#     line = Line()
#     line.m = -2.0
#     line.n = 10.0
#     # Training
#     for i in range(0, 1000):
#         x = randint(-50, 50)
#         y = randint(-50, 50)
#
#         inputs = np.array([x, y])
#         output = network.feed(inputs)
#         network.backpropagate_error(line.diag_line(x, y))
#         network.update_weights()
#
#     # Test
#     sumErrors = 0
#     for i in range(0, 200):
#         x = randint(-50, 50)
#         y = randint(-50, 50)
#
#         raw_output = network.feed(np.array([x,y]))[0]
#         output = 1.0 if raw_output > 0.5 else 0.0
#         expected = line.diag_line(x, y)
#
#         if output != line.diag_line(x, y):
#             sumErrors += 1
#
#         # if (output == 1.0):
#         #     plt.plot(x, y, 'ob')
#         # else:
#         #     plt.plot(x, y, 'xr')
#
#     print("nerror: ", sumErrors)
#     # plt.plot([plt.xlim()[0], plt.xlim()[1]], [line.y(plt.xlim()[0]), line.y(plt.xlim()[1])], ls='--', c='.3')
#
#
# def plot_learning_curve(network):
#
#     lasttr=0
#     lasterror=0
#
#     for tr in range(999, 1000):
#         line = Line()
#         line.m = -1.0
#         line.n = 0.0
#         for i in range(0, tr):
#             x = randint(-50, 50)
#             y = randint(-50, 50)
#
#             inputs = np.array([x, y])
#             output = network.feed(inputs)
#             network.backpropagate_error(line.diag_line(x, y))
#             network.update_weights()
#
#         # Test
#         sumErrors = 0
#         for i in range(0, 100):
#             x = randint(-50, 50)
#             y = randint(-50, 50)
#
#             inputs = np.array([x, y])
#             a = network.feed(inputs)[0]
#             b= line.diag_line(x, y)
#
#             if (network.feed(inputs)[0] != line.diag_line(x, y)):
#                 sumErrors += 1
#
#         if tr>0:
#             plt.plot([lasttr, tr], [lasterror, sumErrors], ls='-', c='.3')
#
#         # print(sumErrors)
#
#         lasterror = sumErrors
#         lasttr=tr

def and_function(x, y):
    return x and y

def or_function(x, y):
    return x or y

def xor_function(x, y):
    return (x and not y) or (not x and y)

def train(network, f):

    # Training
    for i in range(0, 5):

        x = randint(0, 1)
        y = randint(0, 1)

        expected = f(x, y)

        inputs = np.array([x, y])
        output = network.feed(inputs)
        network.backpropagate_error(expected)
        network.update_weights()

    # Test
    sumErrors = 0
    for i in range(0, 1000):
        x = randint(0, 1)
        y = randint(0, 1)

        raw_output = network.feed(np.array([x, y]))[0]
        output = 1.0 if raw_output > 0.5 else 0.0
        expected = f(x, y)

        if output != expected:
            sumErrors += 1

        # if (expected == 1.0):
        #     plt.plot(x, y, 'ob')
        # else:
        #     plt.plot(x, y, 'xr')

    print("nerror: ", sumErrors)

def learning_curve(f, network):

    lasttr=0
    lasterror=0

    realtr = 0
    for tr in range(0, 300):

        for i in range(0, tr):
            x = randint(0, 1)
            y = randint(0, 1)

            expected = f(x,y)

            inputs = np.array([x, y])
            output = network.feed(inputs)
            network.backpropagate_error(expected)
            network.update_weights()

        # print(network)

        # Test
        sumErrors = 0
        totalData = 300
        for i in range(0, totalData):
            x = randint(0, 1)
            y = randint(0, 1)

            raw_output = network.feed(np.array([x, y]))[0]
            output = 1.0 if raw_output > 0.5 else 0.0
            expected = f(x, y)

            if (output != expected ):
                sumErrors += 1

        realtr += tr
        precision = sumErrors / totalData
        print('in ', realtr, ' trainnig got sumerror: ',precision)

        # if realtr>0:
        #     plt.plot([lasttr, realtr], [lasterror, precision], ls='-', c='.3')

        # print(sumErrors)

        lasterror = precision
        lasttr=realtr
    return network

def dummy_test():
    network = NeuronNetwork(2, 0.05)
    # network.add_layer(3)
    # network.add_layer(2)
    # network.add_layer(1)
    # network.layers[0].neurons[0].weights = np.array([0.42, 0.1, 0.6, 0.92])
    # network.layers[0].neurons[0].bias = 0.46
    #
    # network.layers[0].neurons[1].weights = np.array([0.88, 0.73, 0.18, 0.11])
    # network.layers[0].neurons[1].bias = 0.72
    #
    # network.layers[0].neurons[2].weights = np.array([0.55, 0.68, 0.47, 0.52])
    # network.layers[0].neurons[2].bias = 0.08

    network.add_layer(2)
    # network.add_layer(2)
    network.add_layer(1)
    # network.add_layer(1)
    network.layers[0].neurons[0].weights = np.array([1, 1])
    network.layers[0].neurons[0].bias = 0.0

    network.layers[0].neurons[1].weights = np.array([2, 2])
    network.layers[0].neurons[1].bias = 0.0
    #
    # network.layers[1].neurons[0].weights = np.array([2, 2])
    # network.layers[1].neurons[0].bias = 0.0

    # network.layers[1].neurons[1].weights = np.array([2, 2])
    # network.layers[1].neurons[1].bias = 0.0

    network.layers[1].neurons[0].weights = np.array([-1, 0.85])
    network.layers[1].neurons[0].bias = 0.0

    network.feed(np.array([1, 1]))
    print(network)
    network.feed(np.array([1, 0]))
    print(network)
    network.feed(np.array([0, 1]))
    print(network)
    network.feed(np.array([0, 0]))
    print(network)

    return network





network = NeuronNetwork(2, 0.05)
network.add_layer(2)
network.add_layer(1)


# train(network, and_function)
network = learning_curve(xor_function, network)
# plt.show()


# print("network output: ", output)

