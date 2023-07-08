import networkx as nx
import numpy as np
from helperfunctions import *
import time

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

    def add_node(self, i, graph_add_node=True):
        self.node_coords.update({
            i : get_node_coords(i, self.shape)
        })
        if graph_add_node:
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
                        self.graph.add_edge(i, j)
    
    def findlattice(self, removed_nodes, xoffset = 0, yoffset = 0, max_scale = 1):
        """
        Find a raussendorf lattice.

        Returns: cubes: a list containing a cube:

            cube: np.array with shape (19, 3) containing the (x, y, z, scale)
            at [0, :] contains the center of the cube

            n_cubes = the number of cubes found per dimension
        
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

        n_cubes = np.zeros((self.shape[0]//2))

        while scale <= max_scale:
            for c in centers:
                for cube_vec in self.cube:
                    arr = c + cube_vec*scale
                    index = get_node_index(*arr, shape=self.shape)
                    #filter out boundary cases
                    if np.any((arr < 0) | (arr >= self.shape[0])):
                        break
                    #filter out nodes that are measured
                    if removed_nodes[index]:
                        break

                else:
                    cube = np.empty((19, 3))
                    """
                    Format:
                    cube[0, :] = center of the cube
                    cube[:19, :] = coordinates
                    """
                    cube[0, :] = c
                    for i, cube_vec in enumerate(self.cube):
                        cube[i+1, :3] = c + cube_vec*scale
                        #cube[i,  3] = scale 
                    n_cubes[scale-1] += 1
                    cubes.append(cube)
            scale += 1
    
        return cubes, n_cubes


    def findconnectedlattice(self, cubes):
        """
        Extract the data from the numpy array.

        Returns: the graph of centers C
        """
        C = nx.Graph() # C is an object that contains all the linked centers
        for c in cubes:
            center = tuple(c[0, :])
            C.add_node(center)

        for n in C.nodes():
            for n2 in C.nodes():
                length = taxicab_metric(n, n2)
                if length == 2 or length == 3:
                    C.add_edge(n, n2)
        
        connected_cubes = [C.subgraph(c).copy() for c in nx.connected_components(C)]
        return connected_cubes
    
    def findmaxconnectedlattice(self, cubes):
        """
        Extract the data from the numpy array.

        Returns: the graph of centers C
        """
        C = nx.Graph() # C is an object that contains all the linked centers
        for c in cubes:
            center = tuple(c[0, :])
            C.add_node(center)

        for n in C.nodes():
            for n2 in C.nodes():
                length = taxicab_metric(n, n2)
                if length == 2 or length == 3:
                    C.add_edge(n, n2)
        try:
            largest_cc = max(nx.connected_components(C), key=len)
        except ValueError:
            largest_cc = nx.Graph()
        return largest_cc
    
    def connected_cube_to_nodes(self, connected_cubes):
        X = nx.Graph() # X is the same object as C but it contains the actual verticies. 
        
        for node in connected_cubes.nodes():
            for cube_vec in self.cube:
                X.add_node(tuple(node + cube_vec))
        
        for n in X.nodes():
            for n2 in X.nodes():
                if taxicab_metric(n, n2) == 1:
                    X.add_edge(n, n2)
        return X