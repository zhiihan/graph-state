from holes import Holes 
import random
import numpy as np
from helperfunctions import *
import pickle
import multiprocessing as mp
import networkx as nx

shape = 200
samples = 20
p_vec = np.linspace(0.0, 0.25, 50)
plot_data = np.empty((len(p_vec), 2))

for p_index, p in enumerate(p_vec):
    sample_vec = np.zeros(samples)

    for seed in range(samples):
        with open(f'./data/ncubes{p:.4f}shape{shape}sample{seed}', 'rb') as f:
            try:
                ncubes = pickle.load(f)
                sample_vec[seed] = ncubes[0]
            except EOFError:
                print('skipping', p, seed)
            except IndexError:
                print('skipping', p, seed)
    
    sample_vec = np.nan_to_num(sample_vec, neginf=0) 
    print(sample_vec)
    plot_data[p_index, 0] = p
    plot_data[p_index, 1] = np.mean(sample_vec)



import matplotlib.pyplot as plt

plt.title('Unit cubes found')
plt.xlabel('p, probability of losing a node')
plt.ylabel(f'N, cubes found for {shape}')
plt.scatter(plot_data[:, 0], plot_data[:, 1], label=f"Plot data")


y = plot_data[:, 1]
x = plot_data[:, 0]
a, b = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))

plt.plot(x, np.exp(b)* np.exp(a * x), label=f"y = Ae^({a:.1f}x)")
plt.legend()
plt.savefig('unitcubes.png')
print(plot_data)