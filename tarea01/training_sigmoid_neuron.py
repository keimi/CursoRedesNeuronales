from sigmoid_neuron import SigmoidNeuron
from random import randint
import numpy as np
import matplotlib.pyplot as plt


class Line:
    m=0.0
    n=0.0

    def diag_line(self, x, y):
        if self.m*x +self.n > y:
            return 1.0
        else:
            return 0.0

    def y(self, x):
        return self.m*x+self.n



def plot_coordinates():
    w = np.array([1.0, 1.0])
    per = SigmoidNeuron(w, 1.0)
    line = Line()
    line.m = 1.0
    line.n = 50
    # Training
    for i in range(0, 1000):
        x = randint(-50, 50)
        y = randint(-50, 50)

        raw_output = per.feed(np.array([[x, y]]))
        output = 1.0 if raw_output > 0.5 else 0.0
        expected = line.diag_line(x, y)

        if (output != expected):
            w1 = per.weights[0]
            w2 = per.weights[1]
            if (expected == 0.0):
                per.set_weights(np.array([w1 - 0.01 * x, w2 - 0.01 * y]))
            else:
                per.set_weights(np.array([w1 + 0.01 * x, w2 + 0.01 * y]))

    # Test
    sumErrors = 0
    for i in range(0, 200):
        x = randint(-50, 50)
        y = randint(-50, 50)

        raw_output = per.feed(np.array([[x, y]]))
        output = 1.0 if raw_output > 0.5 else 0.0
        expected = line.diag_line(x, y)

        if (output != expected):
            sumErrors += 1

        # if (output == 1.0):
        #     plt.plot(x, y, 'ob')
        # else:
        #     plt.plot(x, y, 'xr')

        if (expected == 1.0):
            plt.plot(x, y, 'ob')
        else:
            plt.plot(x, y, 'xr')

    print('nerror: ', sumErrors)
    plt.plot([plt.xlim()[0], plt.xlim()[1]], [line.y(plt.xlim()[0]), line.y(plt.xlim()[1])], ls='--', c='.3')

def plot_learning_curve():

    lasttr=0
    lasterror=0

    for tr in range(0, 500):
        per = SigmoidNeuron(1.0, 1.0, 1.0)
        line = Line()
        line.m = -1.0
        line.n = 0.0
        for i in range(0, tr):
            x = randint(-50, 50)
            y = randint(-50, 50)

            if (per.process_input(x, y) != line.diag_line(x, y)):
                if (line.diag_line(x, y) == 0):
                    per.set_weights(per.w1 - 0.01 * x, per.w2 - 0.01 * y)
                else:
                    per.set_weights(per.w1 + 0.01 * x, per.w2 + 0.01 * y)

        # Test
        sumErrors = 0
        for i in range(0, 100):
            x = randint(-50, 50)
            y = randint(-50, 50)

            if (per.process_input(x, y) != line.diag_line(x, y)):
                sumErrors += 1

        if tr>0:
            plt.plot([lasttr, tr], [lasterror, sumErrors], ls='-', c='.3')

        # print(sumErrors)

        lasterror = sumErrors
        lasttr=tr



# plt.subplot(1,2,1)
plot_coordinates()

# plt.subplot(1,2,2)
# plot_learning_curve()
plt.show()






# plot learning curve




# print("Number of errors: " + str(sumErrors) + ' Perceptron Weights: ' + str(per.w1) + ' ' + str(per.w2))

