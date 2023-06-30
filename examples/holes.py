import networkx as nx
import numpy as np
from helperfunctions import *

class Holes:
    def __init__(self, shape):
        self.shape = shape
        self.node_coords = {}
        self.graph = nx.Graph()

    def get_node_index(self, x, y, z):
        return x + y * self.shape[1] + z * self.shape[1] * self.shape[2]

    def get_node_coords(self, i):
        index_x = i % self.shape[0]
        index_y = (i // self.shape[0]) % self.shape[1]
        index_z = (i // (self.shape[0] * self.shape[1])) % self.shape[2]
        return np.array([index_x, index_y, index_z])

    def add_node(self, i):
        self.node_coords.update({
            i : get_node_coords(i, self.shape)
        })
        self.graph.add_node(i)

    def are_nodes_connected(self, node1, node2):
        x1 = np.array(self.node_coords[node1])
        x2 = np.array(self.node_coords[node2])

        if np.sum(np.abs(x1 - x2)) == 1:
            return True
        else:
            return False

    def add_edges(self):
        for n in self.graph.nodes:
            for n2 in self.graph.nodes:
                if self.are_nodes_connected(n, n2):
                    self.graph.add_edge(n, n2)
        self.double_hole()

    def to_networkx(self):
        self.add_edges()
        return self.graph

    def return_plot_data(self):
        x_nodes = [self.node_coords[j][0] for j in self.graph.nodes] # x-coordinates of nodes
        y_nodes = [self.node_coords[j][1] for j in self.graph.nodes] # y-coordinates
        z_nodes = [self.node_coords[j][2] for j in self.graph.nodes] # z-coordinates

        #we need to create lists that contain the starting and ending coordinates of each edge.
        x_edges=[]
        y_edges=[]
        z_edges=[]

        #need to fill these with all of the coordinates
        for edge in self.graph.edges:
            #format: [beginning,ending,None]
            x_coords = [self.node_coords[edge[0]][0],self.node_coords[edge[1]][0],None]
            x_edges += x_coords

            y_coords = [self.node_coords[edge[0]][1],self.node_coords[edge[1]][1],None]
            y_edges += y_coords

            z_coords = [self.node_coords[edge[0]][2],self.node_coords[edge[1]][2],None]
            z_edges += z_coords
        
        return [x_nodes, y_nodes, z_nodes], [x_edges, y_edges, z_edges]

    def double_hole(self):
        """
        Check if a hole is a double hole.

        Input: holes object
        Output: 
        """

        #self.double_holes = nx.Graph()

        for i in self.node_coords.keys():
            for j in self.node_coords.keys():
                x_diff = np.abs(np.array(self.node_coords[i]) - np.array(self.node_coords[j]))
                if np.sum(x_diff) == 2:
                    if not ((x_diff[0] == 2) or (x_diff[1] == 2) or (x_diff[2] == 2)):
                        #self.double_holes.add_node(tuple(h))
                        #self.double_holes.add_node(tuple(i))
                        #self.double_holes.add_edge(tuple(h), tuple(i))
                        self.graph.add_edge(i, j)
        #print('doubleholes at ', self.double_holes.edges)
    
    def findlattice(self, removed_nodes, xoffset = 0, yoffset = 0):
        """
        Find a raussendorf lattice.
        """
        
        self.cube = [np.array([0, -1, -1]),
                np.array([-1, 0, -1]),
                np.array([0, 0, -1]),
                np.array([0, 1, -1]),
                np.array([1, 0, -1]),
                np.array([-1, -1, 0]),
                np.array([0, -1, 0]),
                np.array([-1, 0, 0]),
                np.array([-1, 1, 0]),
                np.array([0, 1, 0]),
                np.array([1, 1, 0]),
                np.array([1, 0, 0]),
                np.array([1, -1, 0]),
                np.array([0, -1, 1]),
                np.array([-1, 0, 1]),
                np.array([0, 0, 1]),
                np.array([1, 0, 1]),
                np.array([0, 1, 1])]
        
        scale = 1
        cubes = []
        centers = [np.array([x, y, z]) for z in range(self.shape[2]) for y in range(self.shape[1]) for x in range(self.shape[0])
                if ((x + xoffset) % 2 == z % 2) and ((y + yoffset) % 2 == z % 2)]

        cubes_scales = np.zeros((self.shape[0]//2))

        while scale < self.shape[0]:
            for c in centers:
                for cube_node in self.cube:
                    arr = c + cube_node*scale
                    index = get_node_index(*arr, shape=self.shape)
                    #filter out nodes that are measured
                    if (index in removed_nodes):
                        break
                    #filter out boundary cases
                    if (np.any(arr <= 0)) or (np.any(arr >= self.shape[0])):
                        break
                else:
                    cube = np.empty((18, 4))
                    for i, cube_node in enumerate(self.cube):
                        cube[i, :3] = c + cube_node*scale
                        cube[i,  3] = scale 
                    cubes_scales[scale-1] += 1
                    cubes.append(cube)
                    #print(f"scale = {scale}, center = {c}")
            scale += 1
    
        return cubes, cubes_scales
    

    def findlatticefast(self, removed_nodes, xoffset = 0, yoffset = 0):
        """
        Find a raussendorf lattice.
        """
        
        self.cube = [np.array([0, -1, -1]),
                np.array([-1, 0, -1]),
                np.array([0, 0, -1]),
                np.array([0, 1, -1]),
                np.array([1, 0, -1]),
                np.array([-1, -1, 0]),
                np.array([0, -1, 0]),
                np.array([-1, 0, 0]),
                np.array([-1, 1, 0]),
                np.array([0, 1, 0]),
                np.array([1, 1, 0]),
                np.array([1, 0, 0]),
                np.array([1, -1, 0]),
                np.array([0, -1, 1]),
                np.array([-1, 0, 1]),
                np.array([0, 0, 1]),
                np.array([1, 0, 1]),
                np.array([0, 1, 1])]
        
        scale = 1
        centers = [np.array([x, y, z]) for z in range(self.shape[2]) for y in range(self.shape[1]) for x in range(self.shape[0])
                if ((x + xoffset) % 2 == z % 2) and ((y + yoffset) % 2 == z % 2)]
        print(len(centers), 'centers')
        cubes_scales = np.zeros((self.shape[0]//2), dtype=int)

        removed_nodes_set = set(removed_nodes)

        #while scale < self.shape[0]:
        t = 0
        import time 
        time_delta = time.time()
        while scale < (self.shape[0]//2):
            t = 0
            for c in centers:
                for cube_node in self.cube:
                    arr = c + cube_node*scale
                    index = get_node_index(*arr, shape=self.shape)
                    #filter out nodes that are measured
                    if index in removed_nodes_set:
                        break
                    #filter out boundary cases
                    if (np.any(arr <= 0)) or (np.any(arr >= self.shape[0])):
                        break
                else:
                    #append the size of the cube for now
                    cubes_scales[scale - 1] += 1
                if t % 1000 == 0:
                    print(f"{np.sum(cubes_scales)}, found, {t/len(centers)*100}% finished, scale = {scale/(self.shape[0]//2)*100}%")
                    print(time.time() - time_delta)
                    time_delta = time.time()
                t += 1
            scale += 2
            
        return cubes_scales
