from grid import Grid
from holes import Holes 
import random
import numpy as np
from helperfunctions import *

# Global constants

xoffset = 0
yoffset = 0


def reset_seed(seed, shape):
    """
    Randomly measure qubits.
    """
    global D, removed_nodes
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


def algorithm1(shape):
    holes = D.graph.nodes
    hole_locations = np.zeros(4)
    global xoffset, yoffset

    #counting where the holes are
    for h in holes:
        nx, ny, nz = get_node_coords(h, shape)
        for yoffset in range(2):
            for xoffset in range(2):
                if ((nx + xoffset) % 2 == nz % 2) and ((ny + yoffset) % 2 == nz % 2):
                    hole_locations[xoffset+yoffset*2] += 1
    
    xoffset = np.argmax(hole_locations) // 2
    yoffset = np.argmax(hole_locations) % 2

    #print(xoffset, yoffset)

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                if ((x + xoffset) % 2 == z % 2) and ((y + yoffset) % 2 == z % 2):
                    i = get_node_index(x, y, z, shape)
                    removed_nodes.add(i)
                    #G.handle_measurements(i, 'Z')
                    
                    #move_list.append([i, 'Z']) 


import matplotlib.pyplot as plt
import time

shape = [200, 200, 200]
samples = 1
n_cubes = np.empty((25, shape[0]//2, samples))
p_vec = np.linspace(0.1, 0.25, 25)
D = Holes(shape) # holes
start = time.time()
for index_seed, seed in enumerate(range(samples)):
    for index_p, p in enumerate(p_vec):
        reset_seed(seed, shape)
        print('done reset seed')
        algorithm1(shape)
        print('done alg')
        cube_scales = D.findlatticefast(removed_nodes, xoffset=xoffset, yoffset=yoffset)
        print('done search')
        n_cubes[index_p, :, index_seed] = cube_scales

        print(f'{np.sum(n_cubes[index_p, :, index_seed], dtype=int)} Raussendorf Latticies found for p = {p}, shape = {shape}, cube_dim = {cube_scales}')
        end1loop = time.time()
        print((end1loop-start)/60, 'mins = 1 loop time ')
print((time.time() - start)/60, 'mins = time for 25 points')
#np.save('data.npy', n_cubes)
print(n_cubes.shape)
n_cubes_avg = np.mean(n_cubes, axis=2)

plt.figure()
plt.scatter(p_vec, n_cubes_avg[:, 0], label = f'shape = {shape}, cubesize={1}')
plt.scatter(p_vec, n_cubes_avg[:, 0] + n_cubes_avg[:, 2], label = f'shape = {shape}, cubesize={1, 3}')
plt.scatter(p_vec, n_cubes_avg[:, 0] + n_cubes_avg[:, 2] + n_cubes_avg[:, 4], label = f'shape = {shape}, cubesize={1, 3, 5}')
plt.xlabel('p')
plt.title('Number of Raussendorf Lattices vs. p')
plt.ylabel('N')
plt.legend()

plt.savefig(f'probs50.png')

for i in [4, 12, 20]:
    plt.figure()
    plt.scatter(range(shape[0]//2), n_cubes_avg[i, :], label = f'p = {p_vec[i]}, shape={shape}')
    plt.xlabel('Lattice sizes')
    plt.title('Distribution of lattice sizes')
    plt.ylabel('Number of Raussendorf Lattices count')
    plt.legend()
    plt.savefig(f'hist{i}.png')

plt.figure()
plt.scatter(p_vec, n_cubes_avg[:, 0], label = f'shape = {shape}, cubesize={1}')
plt.scatter(p_vec, n_cubes_avg[:, 2], label = f'shape = {shape}, cubesize={3}')
plt.scatter(p_vec, n_cubes_avg[:, 4], label = f'shape = {shape}, cubesize={5}')
plt.xlabel('p')
plt.title('Number of Raussendorf Lattices vs. p')
plt.ylabel('N')
plt.legend()

plt.savefig(f'probs50_separate.png')