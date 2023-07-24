from holes import Holes 
import numpy as np
from helperfunctions import *
import pickle
import networkx as nx
import multiprocessing as mp

import matplotlib.pyplot as plt

shape = [100, 100, 100]
samples = 10
p_vec = np.linspace(0.0, 0.40, 50)
plot_data = np.empty((len(p_vec), 2))
import multiprocessing as mp
import time
cpu_cores = 8
input_vec = [0, 1, 2, 3, 5, 10, 30, 100]





# original code
def main(input_vec):
    rounds = input_vec

    for p_index, p in enumerate(p_vec):

        sample_vec = np.zeros(samples)

        for seed in range(samples):
            with open(f'./data/cc{p:.4f}shape{shape[0]}{shape[1]}{shape[2]}sample{seed}round{rounds}', 'rb') as f:
                try:
                    cc = pickle.load(f)
                    low = np.array([np.inf, np.inf, np.inf])
                    high = np.zeros(3)
                    for n in cc:
                        low = np.minimum(low, np.array(n))
                        high = np.maximum(high, n)
                    percol_dist = high[0]-low[0]

                    sample_vec[seed] = percol_dist
                    
                    if percol_dist >= shape[0] - 3:
                        print(f'dist={percol_dist}, p={p}, seed={seed}, percolates, round={rounds}')
                    else:
                        print(percol_dist, p, seed, 'does not percolate')
                except EOFError:
                    print('skipping', p, seed)
                except IndexError:
                    print('skipping', p, seed)
        
        sample_vec = np.nan_to_num(sample_vec, neginf=0).astype(int)
        print(sample_vec)
        plot_data[p_index, 0] = p
        plot_data[p_index, 1] = np.mean(sample_vec)
    return (rounds, plot_data)



if __name__ == "__main__":
    start = time.time()
    print(input_vec)
    pool = mp.Pool(processes=cpu_cores)
    results = pool.map(main, input_vec)
    pool.close()
    pool.join()

    for i in results:
        rounds, plot_data = i
        #plt.figure()
        plt.title('x_max - x_min in the largest connected subgraph')
        plt.xlabel('p, probability of losing a node')
        plt.ylabel('x_max - x_min')
        if rounds < 5:
            plt.plot(plot_data[:, 0], plot_data[:, 1], label=f'r = {rounds+1} rounds, shape = {shape}')
        else:
            plt.plot(plot_data[:, 0], plot_data[:, 1], label=f'r = {rounds} rounds, shape = {shape}')
        plt.legend()
    plt.savefig(f'percol{shape[0]}round{rounds}combined.png')