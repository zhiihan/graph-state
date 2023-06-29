from graph_state import GraphState
from collections import Counter
import numpy as np
import networkx as nx

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

    def get_node_index(self, x, y, z):
        return x + y * self.shape[1] + z * self.shape[1] * self.shape[2]

    def generate_cube_edges(self):
        # Generate edges along the height
        for z in range(self.shape[2]):
            for y in range(self.shape[1]):
                for x in range(self.shape[0] - 1):
                    start_node = self.get_node_index(x, y, z)
                    end_node = self.get_node_index(x + 1, y, z)
                    self.edges.append((start_node, end_node))

        # Generate edges along the width
        for z in range(self.shape[2]):
            for y in range(self.shape[1] - 1):
                for x in range(self.shape[0]):
                    start_node = self.get_node_index(x, y, z)
                    end_node = self.get_node_index(x, y + 1, z)
                    self.edges.append((start_node, end_node))

        # Generate edges along the length
        for z in range(self.shape[2] - 1):
            for y in range(self.shape[1]):
                for x in range(self.shape[0]):
                    start_node = self.get_node_index(x, y, z)
                    end_node = self.get_node_index(x, y, z + 1)
                    self.edges.append((start_node, end_node))

    def generate_node_coords(self):
        """
        Get node coordinates.
        """
        self.node_coords = {}

        for z in range(self.shape[2]):
            for y in range(self.shape[1]):
                for x in range(self.shape[0]):
                    self.node_coords.update({
                        self.get_node_index(x, y, z) : np.array([x, y, z])
                    })      

    def damage_grid(self, p, seed=None):
        np.random.seed(seed=seed)
        # p is the probability of losing a qubit

        for i in range(self.shape[0]*self.shape[1]*self.shape[2]):
            if np.random.random() < p:
                self.measure(i)
                self.removed_nodes.append(i)

    def adjaencyMatrix(self):
        return nx.to_numpy_array(self.to_networkx())

    def handle_measurements(self, i, basis):
        self.measure(i, basis=basis)        
        

