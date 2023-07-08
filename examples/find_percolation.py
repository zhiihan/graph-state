from holes import Holes 
import random
import numpy as np
from helperfunctions import *
import pickle
import multiprocessing as mp
import networkx as nx

shape = 15
samples = 5
p_vec = np.linspace(0.0, 0.25, 25)
input_vec = [(p, s) for p in p_vec for s in range(samples)]
plot_data = np.empty((samples*len(p_vec), 2))

for p_index, p in enumerate(p_vec):
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
                plot_data[p_index*samples + seed, 0] = p
                plot_data[p_index*samples + seed, 1] = percol_dist
                if percol_dist >= shape - 3:
                    print(percol_dist, p, seed, 'percolates')
                else:
                    print(percol_dist, p, seed, 'does not percolate')
            except EOFError:
                print('skipping', p, seed)
            except IndexError:
                print('skipping', p, seed)

plot_data = np.nan_to_num(plot_data, neginf=0) 
import matplotlib.pyplot as plt

plt.scatter(plot_data[:, 0], plot_data[:, 1])
plt.savefig('percol.png')
print(plot_data)