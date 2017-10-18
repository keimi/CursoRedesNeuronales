import random
import numpy as np
from sklearn.preprocessing  import normalize



class BitSequence(object):
    step=0
    N=10
    n=6
    mut_rate =0.3

    def __init__(self):
        self.answer = np.array([random.randint(0,1) for i in range(self.n) ])

    def make_population(self):
        population = np.array([[random.randint(0,1) for i in range(self.n) ] for j in range(self.N)])
        return population

    def evaluate_fitness(self, population):
        fitness = np.array([[np.sum(np.logical_not(np.logical_xor(a, self.answer))) for a in population]])

        ret = np.where(fitness==6)
        return ret, fitness

    def make_selection(self, fitness, population):
        fitness_norm = normalize(fitness.astype(float), norm='l1')
        idx = (np.argsort(fitness_norm)[0])[::-1][:self.N]
        fitness_sort = (fitness_norm[0])[idx]
        fitness_accum = np.add.accumulate(fitness_sort)

        n_parents =0
        while(n_parents <= 2):
            R = random.uniform(0.1, 1.0)

            parents = np.array([])

            for i in range(self.N):

                if(i==0):
                    parents = population[idx[i]][:]
                else:
                    new_p = population[idx[i]][:]

                    p_exist = False
                    for p in parents:
                        if(np.array_equal(new_p,p)):
                            p_exist = True
                            break

                    if p_exist:
                        continue

                    parents = np.vstack((parents, new_p))

                if R < fitness_accum[i]:
                    break



            n_parents = int(parents.size / 6)

        print('parents\n' ,parents)
        print('answer: ', self.answer)
        return parents


    def reproduce(self, parents):

        new_population = np.ones((self.N, self.n))

        n_parents = int(parents.size /6)
        # print('parents ', parents)

        for i in range(self.N):
            p1 = random.randint(0, n_parents-1)
            p2 = p1
            while(p1==p2):
                p2 = random.randint(0, n_parents - 1)

            parent1 = parents[p1]
            parent2 = parents[p2]

            div = random.randint(1, self.n-1)

            child = np.hstack( (parent1[0:div], parent2[div:self.n]) )

            child = np.array([ (i if (random.uniform(0,1) <self.mut_rate) else int(not(i))) for i in child ])

            new_population[i][:] = child

        return new_population








b = BitSequence()
# print(b.answer)
population = b.make_population()
index, fitness = b.evaluate_fitness(population)
it =1

while (np.size(index) == 0):
    it+=1
    parents = b.make_selection(fitness, population)
    population = b.reproduce(parents)
    index, fitness = b.evaluate_fitness(population)

print('vector found at iteration', it)
print('anwser: ', b.answer, ' found: ', population[index[1][0]])

# print (b.population)

# print(b.fitness_norm)
# if(np.size(index) == 0):
#     print('not found')
# else:
#     print('found in ', index)
#
#
# print(new_pop)



