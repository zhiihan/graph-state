from holes import Holes 
import random
import numpy as np
from helperfunctions import *

def reset_seed(p, seed, shape):
    """
    Randomly measure qubits.
    """
    D = Holes(shape)
    removed_nodes = set()

    random.seed(int(seed))
    # p is the probability of losing a qubit

    measurementChoice = 'Z'
    for i in range(shape[0]*shape[1]*shape[2]):
        if random.random() < p:
            removed_nodes.add(i)
            D.add_node(i, graph_add_node=False)
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
                    removed_nodes.add(i) 
    
    return xoffset, yoffset

def main(p):
    start = time.time()

    seed = 1

    D, removed_nodes = reset_seed(p, seed, shape)
    xoffset, yoffset = algorithm1(D, removed_nodes, shape)
    cube_scales = D.findlattice(removed_nodes, xoffset=xoffset, yoffset=yoffset, p=p)

    end1loop = time.time()
    print((end1loop-start)/60, 'mins = 1 loop time ')
    return cube_scales



import matplotlib.pyplot as plt
import time
import multiprocessing as mp

cpu_cores = 25

shape = [1000, 1000, 1000]
samples = 1
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

    n_cubes = np.vstack(results)
        
    print(time.time() - start)

    np.save('data.npy', n_cubes)
    print(n_cubes.shape, p_vec.shape)

    plt.figure()
    plt.scatter(p_vec, n_cubes[:, 0], label = f'shape = {shape}, cubesize={1}')
    plt.scatter(p_vec, n_cubes[:, 0] + n_cubes[:, 2], label = f'shape = {shape}, cubesize={1, 3}')
    plt.xlabel('p')
    plt.title('Number of Raussendorf Lattices vs. p')
    plt.ylabel('N')
    plt.legend()

    plt.savefig(f'probs{shape[0]}.png')

    plt.figure()
    plt.scatter(p_vec, n_cubes[:, 0], label = f'shape = {shape}, cubesize={1}')
    plt.scatter(p_vec, n_cubes[:, 2], label = f'shape = {shape}, cubesize={3}')
    plt.xlabel('p')
    plt.title('Number of Raussendorf Lattices vs. p')
    plt.ylabel('N')
    plt.legend()

    plt.savefig(f'probs{shape[0]}_separate.png')

