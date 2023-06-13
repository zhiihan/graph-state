from graph_state import GraphState
from collections import Counter
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5agg")

class Grid(GraphState):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(self.shape[0]*self.shape[1]*self.shape[2])
        self.edges = []
        self.removed = []
        self.get_node_coords()
        self.generate_cube_edges()

        for i in range(self.shape[0]*self.shape[1]*self.shape[2]):
            self.h(i)
        
        for e in self.edges:
            self.add_edge(*e)

        self.G = self.to_networkx()
        

    def get_node_index(self, x, y, z):
        return x + y * self.shape[1] + z * self.shape[1] * self.shape[2]

    def generate_cube_edges(self):
        num_nodes = self.shape[0]*self.shape[1]*self.shape[2]
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

    def get_node_coords(self):
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

    def damage_grid(self, p):
        # p is the probability of losing a qubit

        for i in range(self.shape[0]*self.shape[1]):
            if random.random() < p:
                self.measure(i)
                self.removed.append(i)

    def draw(self):
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('button_press_event', self.onClick)
        
        self.G = self.to_networkx()
        self.pos_nodes = {i: ((i%self.shape[1]), -(i//self.shape[1])) for i in self.G.nodes()}

        for i in self.removed:
            self.G.remove_node(i)

        nx.draw(self.G, self.pos_nodes, with_labels=True)

        pos_attrs = {}
        for node, coords in self.pos_nodes.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)

        node_attrs = nx.get_node_attributes(self.G, 'vop')
        custom_node_attrs = {}
        for node, attr in node_attrs.items():
            custom_node_attrs[node] = attr

        nx.draw_networkx_labels(self.G, pos_attrs, labels=custom_node_attrs)
        plt.show()

    def onClick(self, event):
        (x, y) = (event.xdata, event.ydata)

        for i in self.pos_nodes:
            distance = pow(x-self.pos_nodes[i][0],2)+pow(y-self.pos_nodes[i][1],2)
            if distance < 0.1 and i not in self.removed:
                self.removed.append(i)
                print(i, self.removed) 
                if event.button == 1:
                    print('lmb', x, y)
                    self.measure(i)
                if event.button == 3:
                    print('rmb', x, y)
                    self.measure(i, basis='Y')
        self.refreshGraph()

    def refreshGraph(self):
        plt.clf()
        self.draw()

    def adjaencyMatrix(self):
        return nx.to_numpy_array(self.to_networkx())
