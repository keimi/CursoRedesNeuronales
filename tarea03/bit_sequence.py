import random
import numpy as np
import matplotlib.pyplot as plt
from genetic import Genetic
from genetic import Fit
from genetic import FitFloat

def binary_bit_seq_exp():
    vec_size = 1000
    pop_size = 100
    max_generation = 3000
    answer = np.array([random.randint(0, 1) for i in range(vec_size)])

    fit_class = Fit(answer)

    bit_seq = Genetic(fitness_class=fit_class, gene_size=vec_size, population_size=pop_size, mut_rate=0.1, max_fitness=vec_size, n_pool=4)
    fitest, best_vector, mean_vector = bit_seq.evolve(max_generation)
    max_generation = best_vector.size

    print('vector found at iteration', max_generation)
    print('anwser: ', answer, ' found: ',fitest)
    plt.plot(np.array([i for i in range(max_generation)]), best_vector)
    plt.plot(np.array([i for i in range(max_generation)]), mean_vector, '-r')
    plt.show()

def float_vector_exp():
    vec_size = 10
    pop_size = 1000
    max_generation = 100
    answer = np.array([random.uniform(0, 1) for i in range(vec_size)])

    fit_class = FitFloat(answer)
    vec_seq = Genetic(fitness_class=fit_class, gene_size=vec_size, population_size=pop_size, mut_rate=0.1, max_fitness=vec_size, n_pool=4)
    fitest, best_vector, mean_vector = vec_seq.evolve(max_generation)
    max_generation = best_vector.size

    print('vector found at iteration', max_generation)
    print('anwser: ', answer, ' found: ',fitest)
    plt.plot(np.array([i for i in range(max_generation)]), best_vector)
    plt.plot(np.array([i for i in range(max_generation)]), mean_vector, '-r')
    plt.show()



# binary_bit_seq_exp()

float_vector_exp()

# float_vector_exp()

