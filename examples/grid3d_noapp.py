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

def main(p):
    start = time.time()

    seed = 1
    data = np.zeros(samples)
    for i in range(samples):
        D, removed_nodes = reset_seed(p, seed, shape)
        print('done building grid', f'p = {p}, samples={i}/{samples}')
        xoffset, yoffset = algorithm1(D, removed_nodes, shape)
        cubes, n_cubes = D.findlattice(removed_nodes, xoffset=xoffset, yoffset=yoffset)
        print('latticies found', f'p = {p}, samples={i}/{samples}')
        connected_cubes = D.findconnectedlattice(cubes)
        end1loop = time.time()
        print((end1loop-start)/60, 'mins elapsed', f'p = {p}, samples={i}/{samples}')
        data[i] = len(connected_cubes)
    return np.mean(data)



import matplotlib.pyplot as plt
import time
import multiprocessing as mp

cpu_cores = 2

shape = [10, 10, 10]
samples = 5
n_cubes = np.empty((25, shape[0]//2, samples))
p_vec = np.linspace(0.0, 0.25, 25)

def main2(p):
    print(p)
    time.sleep(0.5)
    return np.array([p, 0])


if __name__ == "__main__":
    start = time.time()
    pool = mp.Pool(processes=cpu_cores)
    results = pool.map(main, p_vec)
    pool.close()
    pool.join()

    #n_cubes = np.vstack(results)
    connected_cubes_len = np.array([results])
        
    print(time.time() - start)

    np.save('data_connected_cubes.npy', connected_cubes_len)
    print(connected_cubes_len.shape, p_vec.shape)

    plt.figure()
    plt.scatter(p_vec, connected_cubes_len, label = f'shape = {shape}, cubesize={1}')
    plt.xlabel('p')
    plt.title('Number of connected subgraphs vs. p')
    plt.ylabel('N')
    plt.legend()

    plt.savefig(f'connectedsubgraph{shape[0]}.png')


