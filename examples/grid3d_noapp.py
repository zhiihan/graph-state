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
            G.handle_measurements(i, measurementChoice)
            move_list.append([i, measurementChoice])
            D.add_node(i)
    D.add_edges()
    print(f'Loaded seed : {seed}')

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

    print(xoffset, yoffset)

    for z in range(G.shape[2]):
        for y in range(G.shape[1]):
            for x in range(G.shape[0]):
                if ((x + xoffset) % 2 == z % 2) and ((y + yoffset) % 2 == z % 2):
                    i = G.get_node_index(x, y, z)
                    if i not in removed_nodes:
                        G.handle_measurements(i, 'Z')
                        removed_nodes.append(i)
                        move_list.append([i, 'Z']) 

    print('Ran Alg1')

def findlattice(nclicks, shape=shape):
    #defect_box = D.carve_out_box()
    global measurements_list
    if nclicks == 1:
        measurements_list = D.findlattice(xoffset=xoffset, yoffset=yoffset)
    #double_holes = D.double_hole_remove_nodes()

    #assert len(defect_box) == len(measurements_list)
    print(f'{len(measurements_list)} Raussendorf Latticies found for p = {p}, shape = {shape}')

    """
    if measurements_list:
        reset_grid(nclicks, move_list_reset=False)
        # Measure the previous grid before we did find lattice
        for i, measurementChoice in move_list:
            G.handle_measurements(i, measurementChoice)
            removed_nodes.append(i)

        # Carve out the outer box
        for i in measurements_list[nclicks % len(measurements_list)]:
            if i not in removed_nodes:
                G.handle_measurements(i, 'Z')
                removed_nodes.append(i)
    """
    return 
import matplotlib.pyplot as plt


s = [15, 15, 15]
latticies = np.empty((50, 10))
p_vec = np.linspace(0, 0.25, 50)
for i, seed in enumerate(range(10)):
    for index, p in enumerate(p_vec):
        print(p)
        reset_grid(1, shape=s)
        reset_seed(1, seed, shape=s)
        algorithm1(1, shape=s)
        findlattice(1, shape=s)
        latticies[index, i] = len(measurements_list)
np.save('data.npy', latticies)
latticies_norm = np.mean(latticies, axis=1)

plt.scatter(p_vec, latticies_norm, label = f'cube_size = {s}')
plt.xlabel('p')
plt.title('Number of Raussendorf Lattices vs. p')
plt.ylabel('N')
plt.legend()

plt.savefig(f'probs{i}.png')
