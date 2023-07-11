from holes import Holes 
import random
import numpy as np
from helperfunctions import *

def reset_seed(p, seed, shape):
    """
    Randomly measure qubits.
    """
    D = Holes(shape)
    removed_nodes = np.zeros(shape[0]*shape[1]*shape[2], dtype=bool)

    random.seed(int(seed))
    # p is the probability of losing a qubit

    measurementChoice = 'Z'
    for i in range(shape[0]*shape[1]*shape[2]):
        if random.random() < p:
            removed_nodes[i] = True
            #D.add_node(i, graph_add_node=False)
        if i % 10000000 == 0:
            print(i/(shape[0]*shape[1]*shape[2])*100)
    return D, removed_nodes


def algorithm1(D, removed_nodes, shape):
    holes = D.graph.nodes
    hole_locations = np.zeros(4)

    #counting where the holes are
    for h in holes:
        nx, ny, nz = get_node_coords(h, shape)
        for yoffset in range(2):
            for xoffset in range(2):
                if ((nx + xoffset) % 2 == nz % 2) and ((ny + yoffset) % 2 == nz % 2):
                    hole_locations[xoffset+yoffset*2] += 1
    
    xoffset = np.argmax(hole_locations) // 2
    yoffset = np.argmax(hole_locations) % 2

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if ((x + xoffset) % 2 == z % 2) and ((y + yoffset) % 2 == z % 2):
                    i = get_node_index(x, y, z, shape)
                    removed_nodes[i] = True
    
    return xoffset, yoffset

import pickle

def main(input):
    """
    input = list containing [probability, seed] 
    """
    start = time.time()

    p, seed = input
    
    data = np.zeros(samples)

    D, removed_nodes = reset_seed(p, seed, shape)
    print('done building grid', f'p = {p}, samples={seed}/{samples}')
    xoffset, yoffset = algorithm1(D, removed_nodes, shape)
    cubes, n_cubes = D.findlattice(removed_nodes, xoffset=xoffset, yoffset=yoffset)
    print('latticies found', f'p = {p}, samples={seed}/{samples}')
    
    connected_cubes = D.findconnectedlatticenk(cubes)        
    #with open(f'./data/cubes{p:.2f}sample{seed}', 'wb') as f:
    #    pickle.dump(connected_cubes, f)
    
    #largestcc = D.findmaxconnectedlattice(cubes)
    #with open(f'./data/cc{p:.2f}sample{seed}shape{shape[0]}', 'wb') as f:
    #    pickle.dump(largestcc, f)

    end1loop = time.time()
    print((end1loop-start)/60, 'mins elapsed', f'p = {p}, samples={seed}/{samples}')
    return 



import matplotlib.pyplot as plt
import time

shape = [100, 100, 100]
samples = 1
n_cubes = np.empty((25, shape[0]//2, samples))
p_vec = np.linspace(0.0, 0.25, 25)

input_vec = [(p, s) for p in p_vec for s in range(samples)]

results = []
start = time.time()
for i in input_vec:
    results.append(main(i))
    print((time.time() - start)/60)