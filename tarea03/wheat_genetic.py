import pandas as pd
import numpy as np
from sklearn import preprocessing
from genetic import FitWheat
from genetic import Genetic
from neuron_network import NeuronNetwork
import matplotlib.pyplot as plt

seeds= pd.read_csv('./data/seeds_dataset.txt', sep='\t')
seeds['t2'] = seeds['t1']
seeds['t3'] = seeds['t1']
seeds['t1'] = seeds['t1'].apply(lambda x: 1 if x==1 else 0)
seeds['t2'] = seeds['t2'].apply(lambda x: 1 if x==2 else 0)
seeds['t3'] = seeds['t3'].apply(lambda x: 1 if x==3 else 0)

min_max_scaler = preprocessing.MinMaxScaler()
seeds_scaled = min_max_scaler.fit_transform(np.array(seeds))

pop_size = 150
max_generation = 50

ar = np.array([7, 10, 20, 10, 5, 3])
fit_class = FitWheat(arqu=ar, data=seeds_scaled)
vec_seq = Genetic(fitness_class=fit_class, population_size=pop_size, mut_rate=0.30, max_fitness=210, n_pool=4)
fitest, best_vector, mean_vector = vec_seq.evolve(max_generation)
max_generation = best_vector.size

def create_network(arqu):
    network = NeuronNetwork(arqu[0], 0.5)
    for i in range(1, arqu.size):
        network.add_layer(arqu[i])
    return network

network = create_network(ar)
network.from_list(fitest)

sumTrue = 0
errors = np.array([])
count = 0
for i in range(0, 200):
    count += 1
    inputs = np.array(seeds_scaled[i][0:7])
    raw_output = network.feed(inputs)
    output = [(1.0 if x > 0.5 else 0.0) for x in raw_output.tolist()]
    expected = np.array(seeds_scaled[i][7:10])

    errors = np.append(errors, np.sum(np.abs(expected - raw_output)))
    if output[0] == expected[0] and output[1] == expected[1] and output[2] == expected[2]:
        sumTrue += 1

precision = float(sumTrue) / float(count)
promError = np.mean(np.abs(errors))
print('prec: ', str(precision), 'error: ', str(promError))

print('vector found at iteration', max_generation)
print('fitest one: \n', fitest)
plt.plot(np.array([i for i in range(max_generation)]), best_vector)
plt.plot(np.array([i for i in range(max_generation)]), mean_vector, '-r')
plt.show()