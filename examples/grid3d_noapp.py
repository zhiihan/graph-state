from holes import Holes 
import random
import numpy as np
from helperfunctions import *
import networkx as nx
from grid import Grid

cpu_cores = 2

shape = [20, 20, 200]
samples = 2
p_vec = np.linspace(0.0, 0.35, 50)

def reset_seed(p, seed, shape, removed_nodes, G):
    """
    Randomly measure qubits.
    """
    D = Holes(shape)

    random.seed(int(seed))
    # p is the probability of losing a qubit

    measurementChoice = 'Z'
    for i in range(shape[0]*shape[1]*shape[2]):
        if random.random() < p:
            if removed_nodes[i] == False:
                removed_nodes[i] = True
                D.add_node(i)
                G.handle_measurements(i, 'Z')
        if i % 10000000 == 0:
            print(i/(shape[0]*shape[1]*shape[2])*100)
    return G, D, removed_nodes


def algorithm1(G, D, removed_nodes, shape):
    holes = D.graph.nodes
    hole_locations = np.zeros(8)

    #counting where the holes are
    for h in holes:
        x, y, z = h
        for zoffset in range(2):
            for yoffset in range(2):
                for xoffset in range(2):
                    if ((x + xoffset) % 2 == (z + zoffset) % 2) and ((y + yoffset) % 2 == (z + zoffset) % 2):
                        hole_locations[xoffset+yoffset*2+zoffset*4] += 1
    
    xoffset = np.argmax(hole_locations) % 2
    yoffset = np.argmax(hole_locations) // 2
    zoffset = np.argmax(hole_locations) // 4

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if ((x + xoffset) % 2 == (z + zoffset) % 2) and ((y + yoffset) % 2 == (z + zoffset) % 2):
                    i = get_node_index(x, y, z, shape)
                    removed_nodes[i] = True
                    G.handle_measurements(i, 'Z')
    
    return G, removed_nodes, [xoffset, yoffset, zoffset]

import pickle

def percolation1(G, removed_nodes):
    """check percolation"""
    gnx = G.to_networkx()

    removed_nodes_reshape = removed_nodes.reshape(shape)

    zmax = shape[2]
    
    zeroplane = removed_nodes_reshape[:, :, 0]
    zmaxplane = removed_nodes_reshape[:, :, zmax-1]

    start = np.argwhere(zeroplane == 0) #This is the coordinates of all valid node in z = 0
    end = np.argwhere(zmaxplane == 0) #This is the coordinates of all valid node in z = L


    for index in range(len(end)):
        i = get_node_index(0, 1, 0, shape)
        j = get_node_index(*end[index], zmax-1, shape)
        if nx.has_path(gnx, i, j):
            percolates = True
            break
    else:
        percolates = False

    return percolates



def main(input):
    """
    input = list containing [probability, seed] 
    """
    start = time.time()

    percol = []
    p, seed = input
    pindex = np.argwhere(p_vec == p)[0][0]
    for s in range(seed):
        removed_nodes = np.zeros(shape[0]*shape[1]*shape[2], dtype=bool)
        G = Grid(shape)
        D = Holes(shape)
        #G, removed_nodes, _ = algorithm1(G, D, removed_nodes, shape)

        G, D, removed_nodes = reset_seed(p, s, shape, removed_nodes, G)
        
        percol1 = percolation1(G, removed_nodes)
        print(f'percolates {percol1} p = {p}, samples={s}/{samples}')
        percol.append(percol1)


        with open(f'./datakero/percol{pindex}shape{shape[2]}sample{s}c', 'wb') as f:
            pickle.dump(percol, f)

    for s in range(seed):
        removed_nodes = np.zeros(shape[0]*shape[1]*shape[2], dtype=bool)
        G = Grid(shape)
        D = Holes(shape)
        G, removed_nodes, _ = algorithm1(G, D, removed_nodes, shape)

        G, D, removed_nodes = reset_seed(p, s, shape, removed_nodes, G)
        
        percol1 = percolation1(G, removed_nodes)
        print(f'percolates {percol1} p = {p}, samples={s}/{samples}')
        percol.append(percol1)


        with open(f'./datakero/percol{pindex}shape{shape[2]}sample{s}s', 'wb') as f:
            pickle.dump(percol, f)

    end1loop = time.time()
    print((end1loop-start)/60, 'mins elapsed', f'p = {p}, samples={seed}/{samples}')
    return 



import matplotlib.pyplot as plt
import time
import multiprocessing as mp

input_vec = [(p, s) for p in p_vec for s in range(samples)]

if __name__ == "__main__":
    start = time.time()
    print(input_vec)
    pool = mp.Pool(processes=cpu_cores)
    results = pool.map(main, input_vec)
    pool.close()
    pool.join()

    #n_cubes = np.vstack(results)
    connected_cubes_len = np.array([results])
        
    print((time.time() - start)/60)
    """
    np.save('data_connected_cubes.npy', connected_cubes_len)
    print(connected_cubes_len.shape, p_vec.shape)

    plt.figure()
    plt.scatter(p_vec, connected_cubes_len, label = f'shape = {shape}, cubesize={1}')
    plt.xlabel('p')
    plt.title('Number of connected subgraphs vs. p')
    plt.ylabel('N')
    plt.legend()

    plt.savefig(f'connectedsubgraph{shape[0]}.png')
    """

