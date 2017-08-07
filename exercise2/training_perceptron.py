from perceptron import Perceptron
from random import randint


import numpy as np
import matplotlib.pyplot as plt

def diag_line(x, y ):
    if(x > y):
        return 0
    else:
        return 1


per = Perceptron(1.0, 1.0, 1.0);
# Training
for i in range(0,1000):
    x = randint(-50, 50)
    y = randint(-50, 50)

    if(per.process_input(x,y) != diag_line(x,y)):
        if(diag_line(x,y) == 0):
            per.set_weights(per.w1 - 0.01 * x, per.w2 - 0.01 * y)
        else:
            per.set_weights(per.w1 + 0.01 * x, per.w2 + 0.01 * y)

# Test
sumErrors=0
for i in range(0,200):
    x = randint(-50, 50)
    y = randint(-50, 50)

    if (per.process_input(x, y) != diag_line(x, y)):
        sumErrors += 1

    if( per.process_input(x, y) ==1):
        plt.plot(x,y,'ob')
    else:
        plt.plot(x, y, 'xr')

plt.plot([-50,50],[-50,50], ls='--', c='.3')
plt.show()


print("Number of errors: " + str(sumErrors) + ' Perceptron Weights: ' + str(per.w1) + ' ' + str(per.w2))

