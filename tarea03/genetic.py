import random
import numpy as np
from sklearn.preprocessing  import normalize
import matplotlib.pyplot as plt


class Genetic(object):

    def __init__(self, fitness_class, vector_size=6, population_size=10, mut_rate=0.1, max_steps=0, max_fitness=6):
        # population size
        self.N = population_size
        # vector size
        self.n = vector_size
        # mutation factor
        self.mut_rate = mut_rate
        # max steps in algorithm
        self.max_steps = max_steps
        # max fitness to terminate
        self.max_fitness = max_fitness

        # fitness function takes the charateristics vector and calculate fitness
        self.fitness_class = fitness_class

    def make_population(self, vector_type=0):

        if vector_type == 0:
            population = np.array([[random.randint(0, 1) for i in range(self.n)] for j in range(self.N)])
        else:
            population = np.array([[random.uniform(0, 1) for i in range(self.n)] for j in range(self.N)])
        return population

    def evaluate_fitness(self, population):
        fitness = np.array([[ self.fitness_class.evaluate(a) for a in population]])

        ret = np.where(fitness == self.n)
        return ret, fitness

    def make_selection_trunc(self, fitness, population):
        fitness_norm = normalize(fitness.astype(float), norm='l1')
        idx = (np.argsort(fitness_norm)[0])[::-1][:self.N]
        fitness_sort = (fitness_norm[0])[idx]
        fitness_accum = np.add.accumulate(fitness_sort)
        fitness_accum = np.add.accumulate(fitness_sort)

        parents = np.array([])

        for i in range(self.N):

            # print('parents size: ', parents.size)
            R = random.uniform(0.0, 1.0)

            # print('fit accum: ', fitness_accum[i])
            if fitness_accum[i] < R:
                continue

            if(parents.size == 0):
                parents = population[idx[i]][:]
            else:
                new_p = population[idx[i]][:]
                parents = np.vstack((parents, new_p))

        return parents, fitness[0][idx[0]]

    def make_selection_roulette(self, fitness, population, select_frac=0.1):

        fitness_prob = normalize(fitness.astype(float), norm='l1')
        # print('fitness prob: ', fitness_prob)
        # print('fitness : ', fitness)
        fitness_accum = np.add.accumulate(fitness_prob[0])

        parents = np.array([])
        parents_fit = np.array([])

        for i in range(int(self.N*select_frac)):

            # print('parents size: ', parents.size)
            R = random.uniform(0.0, 1.0)
            for i, val in enumerate(population):
                # print('fit accum: ', fitness_accum[i])
                if R <= fitness_accum[i] and (i==0 or fitness_accum[i-1] < R):
                    # print('chosen: ', i, ' fitness: ', fitness[0][i])
                    if(parents.size == 0):
                        parents = population[i][:]
                        parents_fit = np.array([fitness[0][i]])
                    else:
                        new_p = population[i][:]
                        parents = np.vstack((parents, new_p))
                        parents_fit = np.vstack((parents_fit, np.array([fitness[0][i]])))
                else:
                    continue

        print('parents mean fit: ', np.mean(parents_fit))
        return parents, np.max(fitness[0])

    def reproduce(self, parents):

        new_population = np.ones((self.N, self.n))

        n_parents = int(parents.size /self.n)
        for i in range(self.N):
            p1 = random.randint(0, n_parents-1)
            p2 = p1
            while(p1==p2):
                p2 = random.randint(0, n_parents - 1)

            parent1 = parents[p1]
            parent2 = parents[p2]

            div = random.randint(1, self.n-1)

            child = np.hstack( (parent1[0:div], parent2[div:self.n]) )

            child = np.array([ (i if (random.uniform(0,1) > self.mut_rate) else int(not(i))) for i in child ])

            new_population[i][:] = child


        return new_population

class Fitness(object):

    ref = np.array([])
    fit_func = lambda a, ref: np.sum(np.logical_not(np.logical_xor(a, ref)))

    def __init__(self, ref):
        self.ref = ref

    def set_func(self, fun):
        self.fit_func = fun

    def evaluate(self, vec):
        return self.fit_func(vec, self.ref)


vec_size = 100
pop_size = 1000
answer = np.array([random.randint(0, 1) for i in range(vec_size)])

fit_class = Fitness(answer)
fit_class.fit_func = lambda a, ref: np.sum(np.logical_not(np.logical_xor(a, ref)))

bit_seq = Genetic(fitness_class=fit_class, vector_size=vec_size, population_size=pop_size, mut_rate=0.001)
# print(b.answer)
population = bit_seq.make_population()
index, fitness = bit_seq.evaluate_fitness(population)
mean_fit = np.mean(fitness)
it =1
print('iteration: ', it, ' max fit: ', np.max(fitness), ' mean fit: ', mean_fit)
its = np.array([it])
means = np.array([np.max(fitness)])
# plt.plot(it,mean_fit)


while (np.size(index) == 0 and it < 300):
    it+=1
    parents, max_fit = bit_seq.make_selection_roulette(fitness, population)
    mean_fit = np.mean(fitness)
    population = bit_seq.reproduce(parents)
    index, fitness = bit_seq.evaluate_fitness(population)
    print('iteration: ', it, ' max fit: ', max_fit, ' mean fit: ', mean_fit)
    its = np.vstack((its, it))
    means = np.vstack((means, max_fit))


print('vector found at iteration', it)
if(index[1].size >0):
    print('anwser: ', answer, ' found: ', population[index[1][0]])
plt.plot(its, means)
plt.show()


