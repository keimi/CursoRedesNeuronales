import numpy as np
import matplotlib.pyplot as plt
from genetic import Genetic
from genetic import FitChTree
from chiffres_tree import ChNode

gen_size = 8
pop_size = 100
max_generation = 300
# answer = np.array([random.randint(0, 1) for i in range(vec_size)])

fit_class = FitChTree(347)
ChNode.number_set = [10, 1, 25, 9, 3, 6]

asbtract_tree = Genetic(fitness_class=fit_class, gene_size=gen_size, population_size=pop_size, mut_rate=0.1,
                  max_fitness=0, n_pool=None)
fitest, best_vector, mean_vector = asbtract_tree.evolve(max_generation)
max_generation = best_vector.size

print('vector found at iteration', max_generation)
print(' found: ', fitest)
print('found evaluate: ', fitest.evaluate())
plt.plot(np.array([i for i in range(max_generation)]), best_vector)
# plt.plot(np.array([i for i in range(max_generation)]), mean_vector, '-r')
plt.show()