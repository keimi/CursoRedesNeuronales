import numpy as np
import matplotlib.pyplot as plt
from genetic import Genetic
from genetic import FitAbstractTree

gen_size = 3
pop_size = 100
max_generation = 300
# answer = np.array([random.randint(0, 1) for i in range(vec_size)])

fit_class = FitAbstractTree(None)

asbtract_tree = Genetic(fitness_class=fit_class, gene_size=gen_size, population_size=pop_size, mut_rate=0.01,
                  max_fitness=1E100, n_pool=None)
fitest, best_vector, mean_vector = asbtract_tree.evolve(max_generation)
max_generation = best_vector.size

print('vector found at iteration', max_generation)
print(' found: ', fitest)
line1 = plt.semilogy(np.array([i for i in range(max_generation)]), best_vector, label='best_individual_fitness')
line2 = plt.semilogy(np.array([i for i in range(max_generation)]), mean_vector, '-r', label='mean_fitness')
plt.legend(['best_individual_fitness', 'mean_fitness'])

plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.show()