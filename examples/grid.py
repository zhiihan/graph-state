from graph_state import GraphState
from collections import Counter
import numpy as np
import networkx as nx
from helperfunctions import *

class Grid(GraphState):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(self.shape[0]*self.shape[1]*self.shape[2])

        self.edges = [] 
        self.removed_nodes = []
        self.generate_node_coords()
        self.generate_cube_edges()

        for i in range(self.shape[0]*self.shape[1]*self.shape[2]):
            self.h(i)
        
        for e in self.edges:
            self.add_edge(*e)

    def generate_cube_edges(self):
        edges = []
        nx, ny, nz = self.shape
        num_nodes = nx * ny * nz

        # Generate edges along the x-axis
        for i in range(num_nodes):
            if (i % nx) < (nx - 1):
                edges.append((i, i + 1))

        # Generate edges along the y-axis
        for i in range(num_nodes):
            if (i % (nx * ny)) < (nx * (ny - 1)):
                edges.append((i, i + nx))

        # Generate edges along the z-axis
        for i in range(num_nodes):
            if (i + nx * ny) < num_nodes:
                edges.append((i, i + nx * ny))

        self.edges = edges
        return 

    def generate_node_coords(self):
        """
        Get node coordinates.
        """
        self.node_coords = {}

        for z in range(self.shape[2]):
            for y in range(self.shape[1]):
                for x in range(self.shape[0]):
                    self.node_coords.update({
                        get_node_index(x, y, z, self.shape) : np.array([x, y, z])
                    })      

    def adjaencyMatrix(self):
        return nx.to_numpy_array(self.to_networkx())

    def handle_measurements(self, i, basis):
        
        self.measure(i, basis=basis)        
        

