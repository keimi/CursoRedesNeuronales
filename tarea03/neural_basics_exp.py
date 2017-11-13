import random
import numpy as np
import matplotlib.pyplot as plt
from genetic import Genetic
from genetic import FitXor


def xor_neural_genetic():
    vec_size = 5 # number of neurons in network
    pop_size = 100
    max_generation = 1000

    fit_class = FitXor(np.array([2,6, 4,2,1]))
    vec_seq = Genetic(fitness_class=fit_class, gene_size=vec_size, population_size=pop_size, mut_rate=0.2, max_fitness=4, n_pool=4)
    fitest, best_vector, mean_vector = vec_seq.evolve(max_generation)
    max_generation = best_vector.size

    print('vector found at iteration', max_generation)
    print('fitest one: \n', fitest)
    # print('anwser: ', answer, ' found: ',fitest)
    plt.plot(np.array([i for i in range(max_generation)]), best_vector)
    plt.plot(np.array([i for i in range(max_generation)]), mean_vector, '-r')
    plt.show()

xor_neural_genetic()

# train_n_epochs_with_function(net, xor_function, 20000)
# [prec, error] = test_precision_with_funcction(net, xor_function, 1000)

# network1 = create_2_2_1_network()
# network2 = create_2_2_1_network()
# print(network1.to_nparray())
# print(network2)
# network2.from_nparray(network1.to_nparray())
# print(network2)


