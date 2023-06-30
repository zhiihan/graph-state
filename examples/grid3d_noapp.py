from grid import Grid
from holes import Holes 
import random
import numpy as np
from helperfunctions import *

# Global constants

xoffset = 0
yoffset = 0
shape = [15, 15, 15]
G = Grid(shape) # qubits
D = Holes(shape) # holes
removed_nodes = G.removed_nodes
move_list = [] #local variable containing moves

def reset_grid(input, move_list_reset = True, shape = shape):
    global G
    global removed_nodes
    global move_list
    global D
    G = Grid(shape)
    removed_nodes = []
    D = Holes(shape)
    move_list = []

def reset_seed(nclicks, seed, shape=shape):
    """
    Randomly measure qubits.
    """
    global D
    D = Holes(shape)

    random.seed(int(seed))
    # p is the probability of losing a qubit

    measurementChoice = 'Z'
    #D.add_node(13)
    for i in range(shape[0]*shape[1]*shape[2]):
        if random.random() < p:
            removed_nodes.append(i)
            #G.handle_measurements(i, measurementChoice)
            #move_list.append([i, measurementChoice])
            D.add_node(i)
    #D.add_edges()


def algorithm1(nclicks, shape=shape):
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

    for z in range(G.shape[2]):
        for y in range(G.shape[1]):
            for x in range(G.shape[0]):
                if ((x + xoffset) % 2 == z % 2) and ((y + yoffset) % 2 == z % 2):
                    i = G.get_node_index(x, y, z)
                    removed_nodes.append(i)
                    #G.handle_measurements(i, 'Z')
                    
                    #move_list.append([i, 'Z']) 


import matplotlib.pyplot as plt


s = [50, 50, 50]
samples = 1
n_cubes = np.empty((25, s[0]//2, samples))
p_vec = np.linspace(0, 0.25, 25)
for index_seed, seed in enumerate(range(samples)):
    for index_p, p in enumerate(p_vec):
        reset_grid(1, shape=s)
        reset_seed(1, seed, shape=s)
        print('done reset seed')
        algorithm1(1, shape=s)
        print('done alg')
        cube_scales = D.findlatticefast(removed_nodes, xoffset=xoffset, yoffset=yoffset)
        print('done search')
        n_cubes[index_p, :, index_seed] = cube_scales

        print(f'{np.sum(n_cubes[index_p, :, index_seed], dtype=int)} Raussendorf Latticies found for p = {p}, shape = {s}, cube_dim = {cube_scales}')
        
#np.save('data.npy', n_cubes)
print(n_cubes.shape)
n_cubes_avg = np.mean(n_cubes, axis=2)

plt.figure()
plt.scatter(p_vec, n_cubes_avg[:, 0], label = f'shape = {s}, cubesize={1}')
plt.scatter(p_vec, n_cubes_avg[:, 0] + n_cubes_avg[:, 2], label = f'shape = {s}, cubesize={1, 3}')
plt.scatter(p_vec, n_cubes_avg[:, 0] + n_cubes_avg[:, 2] + n_cubes_avg[:, 4], label = f'shape = {s}, cubesize={1, 3, 5}')
plt.xlabel('p')
plt.title('Number of Raussendorf Lattices vs. p')
plt.ylabel('N')
plt.legend()

plt.savefig(f'probs50.png')

for i in [4, 12, 20]:
    plt.figure()
    plt.scatter(range(s[0]//2), n_cubes_avg[i, :], label = f'p = {p_vec[i]}, shape={s}')
    plt.xlabel('Lattice sizes')
    plt.title('Distribution of lattice sizes')
    plt.ylabel('Number of Raussendorf Lattices count')
    plt.legend()
    plt.savefig(f'hist{i}.png')

plt.figure()
plt.scatter(p_vec, n_cubes_avg[:, 0], label = f'shape = {s}, cubesize={1}')
plt.scatter(p_vec, n_cubes_avg[:, 2], label = f'shape = {s}, cubesize={3}')
plt.scatter(p_vec, n_cubes_avg[:, 4], label = f'shape = {s}, cubesize={5}')
plt.xlabel('p')
plt.title('Number of Raussendorf Lattices vs. p')
plt.ylabel('N')
plt.legend()

plt.savefig(f'probs50_separate.png')