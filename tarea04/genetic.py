import random
import numpy as np
from sklearn.preprocessing  import normalize
from abstract_tree import AbstractTree
from chiffres_tree import ChTree
import multiprocessing as mp


class Genetic(object):

    def __init__(self, fitness_class, gene_size=6, population_size=10, mut_rate=0.1, max_fitness=6, n_pool=1):
        # population size
        self.N = population_size
        # vector size
        self.n = gene_size
        # mutation factor
        self.mut_rate = mut_rate
        # max fitness to terminate
        self.max_fitness = max_fitness
        # fitness function takes the charateristics vector and calculate fitness
        self.fit_class = fitness_class
        # multiprocess pool
        self.pool = mp.Pool(processes=n_pool) if n_pool is not None and  n_pool > 1 else None

    def make_population(self):
        population = [ self.fit_class.random_individual(self.n) for j in range(self.N)]
        return population

    def evaluate_fitness(self, population):
        if self.pool is not None:
            fitness = np.array(self.pool.map(self.fit_class.evaluate, population))
        else:
            fitness = np.array([self.fit_class.evaluate(a) for a in population])
        return fitness

    def make_selection_tournament(self, fitness, population, select_frac=0.1):
        parents = []
        parents_fit = np.array([])
        for i in range(int(self.N * select_frac)):
            best = None
            for j in range(0, 5):
                ind = random.randint(0, self.N -1)
                if best is None or fitness[ind] > fitness[best]:
                    best = ind

            if (len(parents) == 0):
                parents_fit = np.array([fitness[best]])
            else:
                parents_fit = np.vstack((parents_fit, np.array([fitness[best]])))

            parents.append(population[best])

        return parents

    def make_selection_roulette(self, fitness, population, select_frac=0.1):
        min_fit = np.min(fitness.astype(float))
        if(min_fit < 0):
            fitness_prob = normalize(np.array([fitness.astype(float) - min_fit + 0.1]), norm='l1')
        else:
            fitness_prob = normalize(np.array([fitness.astype(float)]), norm='l1')
        fitness_accum = np.add.accumulate(fitness_prob[0])

        parents = []
        parents_fit = np.array([])

        for i in range(int(self.N*select_frac)):

            # print('parents size: ', parents.size)
            R = random.uniform(0.0, 1.0)
            for i, val in enumerate(population):
                # print('fit accum: ', fitness_accum[i])
                if R <= fitness_accum[i] and (i==0 or fitness_accum[i-1] < R):
                    # print('chosen: ', i, ' fitness: ', fitness[0][i])
                    if(len(parents) == 0):
                        parents_fit = np.array([fitness[i]])
                    else:
                        parents_fit = np.vstack((parents_fit, np.array([fitness[i]])))

                    parents.append(val)
                    break
                else:
                    continue

        print('parents mean fit: ', np.mean(parents_fit))
        return parents

    def reproduce(self, parents):

        new_population = []

        n_parents = len(parents)
        for i in range(self.N):
            p1 = random.randint(0, n_parents-1)
            p2 = random.randint(0, n_parents - 1)
            while(p1==p2):
                p2 = random.randint(0, n_parents - 1)

            parent1 = parents[p1]
            parent2 = parents[p2]

            child = self.fit_class.reproduce_parents(parent1, parent2)
            child = self.fit_class.mut_func(child, mut_rate=self.mut_rate)

            new_population.append(child)
        return new_population

    def evolve(self, generations_num):
        population = self.make_population()
        fitest = []
        best_fit = -np.inf

        for it in range(generations_num):
            #########################################
            ## EVALUATE FITNESS
            fitness = self.evaluate_fitness(population)
            # print(fitness)

            mean_fit = np.mean(fitness)
            max_index = np.argmax(fitness)
            max_fit = fitness[max_index]

            if max_fit > best_fit:
                best_fit = max_fit
                fitest = population[max_index]

            ## Check terminal condition
            if(best_fit >= self.max_fitness):
                break

            print('iteration: ', it, ' max fit: ', best_fit, ' mean fit: ', mean_fit)

            if it==0:
                best_vector = np.array([max_fit])
                mean_vector = np.array([mean_fit])
            else:
                best_vector = np.vstack((best_vector, max_fit))
                mean_vector = np.vstack((mean_vector, mean_fit))
            #########################################
            ## MAKE SELECTION
            parents = self.make_selection_tournament(fitness, population, select_frac=0.3)

            #########################################
            ## REPRODUCE
            population = self.reproduce(parents)
            #########################################

        return fitest, best_vector, mean_vector


class Fit(object):
    def __init__(self, ref=np.array([])):
        self.ref = ref

    # Overwrite this function for especific problem
    def random_individual(self, gene_num):
        return np.array([self.random_gene_func(0, 1) for i in range(gene_num)])

    # Overwrite this function for especific problem
    def random_gene_func(self, min, max):
        return np.random.randint(min, max)

    # Overwrite this function for especific problem
    def evaluate(self, vec):
        return np.sum(np.logical_not(np.logical_xor(vec, self.ref)))

    # Overwrite this function for especific problem
    def reproduce_parents(self, p1, p2):
        n = p1.shape[0]
        div = random.randint(1, n - 1)
        return np.concatenate((p1[0:div], p2[div:n]))

    # Overwrite this function for especific problem
    def mut_func(self, individual, mut_rate):
        # Change one gene if mutation is activated
        if (random.uniform(0, 1) <= mut_rate):
            index = random.randint(0, individual.size-1)
            individual[index] = int(not(individual[index]))
        return individual

class FitFloat(Fit):

    def random_gene_func(self, min, max):
        return np.random.uniform(min, max)

    def mut_func(self, individual, mut_rate):
        # Change one gene if mutation is activated
        if (random.uniform(0, 1) <= mut_rate):
            index = random.randint(0, individual.size-1)
            individual[index] = random.uniform(0,1)
        return individual

    def evaluate(self, vec):
        return -np.sum(np.power(vec - self.ref, 2))

from neuron_network import NeuronNetwork

# Genetics Fits For Neuronal Networks
class FitXor(Fit):

    def __init__(self, arqu):
        super().__init__()
        self.arq = arqu

    def random_gene_func(self, min, max, n_weight):
        return np.array([random.uniform(-10.0, 10.0) for i in range(n_weight+1)])

    def random_individual(self, gene_num):
        indv = []
        for j, val in enumerate(self.arq[0:self.arq.size-1]):
            for i in range(self.arq[j+1]):
                indv.append(self.random_gene_func(0, 1, val))
        return indv

    def evaluate(self, vec):
        network = create_network(np.array(self.arq))
        network.from_list(vec)
        fit = 0

        input  = np.array([0, 0])
        output = 1.0 if network.feed(input)[0] > 0.5 else 0.0
        if output == xor_function(input[0], input[1]):
            fit +=1

        input  = np.array([1, 0])
        output = 1.0 if network.feed(input)[0] > 0.5 else 0.0
        if output == xor_function(input[0], input[1]):
            fit +=1

        input = np.array([0, 1])
        output = 1.0 if network.feed(input)[0] > 0.5 else 0.0
        if output == xor_function(input[0], input[1]):
            fit += 1

        input = np.array([1, 1])
        output = 1.0 if network.feed(input)[0] > 0.5 else 0.0
        if output == xor_function(input[0], input[1]):
            fit += 1

        return fit

    def mut_func(self, individual, mut_rate):
        if (random.uniform(0, 1) <= mut_rate):
            index = random.randint(0, len(individual) - 1)
            index2 = random.randint(0, len(individual[index]) - 1)

            # Choose a random weight and negated
            individual[index] [index2] = -individual[index] [index2]

            # Choose a random weight and change it for random
            # individual[index][index2] = random.uniform(-2.0, 2.0)

        return individual

    def reproduce_parents(self, p1, p2):
        child = np.copy(p1)
        for i, gene in enumerate(p1):
            if random.randint(0,1) == 1:
                child[i] = p2[i]
        return child

# Fit class for wheat
class FitWheat(FitXor):

    def __init__(self, arqu, data):
        super().__init__(arqu=arqu)
        self.data = data

    def evaluate(self, vec):
        network = create_network(np.array(self.arq))
        network.from_list(vec)

        error = 0
        sumTrue = 0
        for i in range(210):
            inputs = np.array(self.data[i][0:7])
            raw_output = network.feed(inputs)
            output = [(1.0 if x > 0.5 else 0.0) for x in raw_output.tolist()]
            expected = np.array(self.data[i][7:10])

            error += np.sum(np.abs(expected - raw_output))
            # if output[0] == expected[0]:
            #     sumTrue += 1
            # else:
            #     sumTrue -= 1
            # if output[1] == expected[1]:
            #     sumTrue +=1
            # else:
            #     sumTrue -= 1
            # if output[2] == expected[2]:
            #     sumTrue +=1
            # else:
            #     sumTrue -= 1

            if output[0] == expected[0] and output[1] == expected[1] and output[2] == expected[2]:
                sumTrue += 1

        return sumTrue

    def random_gene_func(self, min, max, n_weight):
        return np.array([random.uniform(-10.0, 10.0) for i in range(n_weight+1)])

    def mut_func(self, individual, mut_rate):
        if (random.uniform(0, 1) <= mut_rate):
        # for i in range(random.randint(0, len(individual) - 1)):
            index = random.randint(0, len(individual) - 1)
            index2 = random.randint(0, len(individual[index]) - 1)

            # Choose a random weight and negated
            individual[index] [index2] = -individual[index] [index2]

            # Scale weight
            # individual[index][index2] = individual[index][index2] * random.uniform(-2.0, 2.0)

                # Choose a random weight and change it for random
                # individual[index][index2] = random.uniform(-2.0, 2.0)

        return individual

# Fit class for abstract tree
class FitAbstractTree(Fit):
    tree=AbstractTree

    def random_individual(self, gene_num):
        # for abstract tree the gene number will be the maximum level available
        return self.tree.random_tree(0.01, 10.0, gene_num)

    def evaluate(self, vec):
        return vec.evaluate()

    def reproduce_parents(self, p1, p2):
        return self.tree.cross_tree(p1,p2)

    def mut_func(self, individual, mut_rate):
        # Change one gene if mutation is activated
        if random.uniform(0, 1) <= mut_rate:
            return self.tree.cross_tree(individual, self.tree.random_tree(0.01, 10.0, 3))
        else:
            return individual

class FitChTree(FitAbstractTree):
    tree=ChTree

    def evaluate(self, vec):
        return -abs( vec.evaluate() - self.ref)


## Other Functions
def create_network(arqu):
    network = NeuronNetwork(arqu[0], 0.5)
    for i in range(1, arqu.size):
        network.add_layer(arqu[i])
    return network

def train_n_epochs_with_function(network, f, n):
    for i in range(0, n):
        x = random.randint(0, 1)
        y = random.randint(0, 1)

        expected = f(x, y)

        inputs = np.array([x, y])
        output = network.feed(inputs)
        network.backpropagate_error(expected)
        network.update_weights()
    return network

def xor_function(x, y):
    return int((x and not y) or (not x and y))

def test_precision_with_function(network, f, n):
    sumTrue = 0
    for i in range(0, n):
        x = random.randint(0, 1)
        y = random.randint(0, 1)

        raw_output = network.feed(np.array([x, y]))[0]
        output = 1.0 if raw_output > 0.5 else 0.0
        expected = f(x, y)
        if output == expected:
            sumTrue += 1

    return sumTrue



#
# vec = np.array( [[-0.6346307,  -1.2013637,  -0.82618694],
#  [-1.38513191,  0.73927323, -0.73247287],
#  [ 0.99942756,  0.39981322,  0.51377254],
#  [-0.09175196,  0.259219,    0.28817243],
#  [-0.95704987,  1.08841506,  0.04107616]])

# vec = np.array([[6.266287706822468, 6.2508760190124475, -2.73248187321048], [4.318093260635528, 4.291841995875093, -6.605775223442097], [8.778316049344214, -9.456038610360649, -4.029161400294834]])
#
# vec = vec/10.0
# print(vec)
#
# print(xor_function(0.0, 0.0))
# print(xor_function(1.0, 0.0))
# print(xor_function(0.0, 1.0))
# print(xor_function(1.0, 1.0))

# fitclass = FitXor()
# print(fitclass.evaluate(vec))

# network1 = create_network(np.array([2,2,2,1]))
# network2 = create_network(np.array([2,2,2,1]))
# print(network)

# print(network2.to_list())
# list = network2.to_list()
# network1.from_list(list)
#
# print(network1)
# network2 = create_2_2_2_1_network()
# print(network1.to_nparray())
# print(network2)
# network2.from_nparray(network1.to_nparray())
# print(network2)