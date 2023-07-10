from holes import Holes 
import random
import numpy as np
from helperfunctions import *
import pickle
import multiprocessing as mp
import networkx as nx

shape = 50
samples = 3
p_vec = np.linspace(0.0, 0.25, 25)
plot_data = np.empty((len(p_vec), 2))

for p_index, p in enumerate(p_vec):

    sample_vec = np.zeros(samples)

    for seed in range(samples):
        with open(f'./data/cc{p:.2f}sample{seed}shape{shape}', 'rb') as f:
            try:
                cc = pickle.load(f)
                low = np.array([np.inf, np.inf, np.inf])
                high = np.zeros(3)
                for n in cc:
                    low = np.minimum(low, np.array(n))
                    high = np.maximum(high, n)
                percol_dist = high[0]-low[0]

                sample_vec[seed] = percol_dist
                
                if percol_dist >= shape - 3:
                    print(percol_dist, p, seed, 'percolates')
                else:
                    print(percol_dist, p, seed, 'does not percolate')
            except EOFError:
                print('skipping', p, seed)
            except IndexError:
                print('skipping', p, seed)
    
    sample_vec = np.nan_to_num(sample_vec, neginf=0) 
    print(sample_vec)
    plot_data[p_index, 0] = p
    plot_data[p_index, 1] = np.mean(sample_vec)



import matplotlib.pyplot as plt

plt.title('x_max - x_min in the largest connected subgraph')
plt.xlabel('p, probability of losing a node')
plt.ylabel('x_max - x_min')
plt.scatter(plot_data[:, 0], plot_data[:, 1])
plt.savefig('percol.png')
print(plot_data)