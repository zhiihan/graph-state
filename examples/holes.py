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
        self.add_edges()

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

        self.double_holes = nx.Graph()

        for h in self.node_coords.values():
            for i in self.node_coords.values():
                x_diff = np.abs(np.array(i) - np.array(h))
                if np.sum(x_diff) == 2:
                    if not ((x_diff[0] == 2) or (x_diff[1] == 2) or (x_diff[2] == 2)):
                        self.double_holes.add_node(tuple(h))
                        self.double_holes.add_node(tuple(i))
                        self.double_holes.add_edge(tuple(h), tuple(i))
        print('doubleholes at ', self.double_holes.edges)
                
    def double_hole_remove_nodes(self):
        """
        Remove nodes from double holes.
        """
        plan_to_measure = []
        for edge in self.double_holes.edges:
            diff = np.array(edge[1]) - np.array(edge[0])
            if diff[0] != 0:
                plan_to_measure.append(np.array(edge[0]) + np.array([diff[0], 0, 0]))
            if diff[1] != 0:
                plan_to_measure.append(np.array(edge[0]) + np.array([0, diff[1], 0]))
            if diff[2] != 0:
                plan_to_measure.append(np.array(edge[0]) + np.array([0, 0, diff[2]]))

        plan_to_measure_index = [get_node_index(*i, shape=self.shape) for i in plan_to_measure]
        return plan_to_measure_index

        
#D.add_node(0, 0, 0)
#D.add_node(0, 0, 1)
#D.add_node(0, 0, 2)
#print(D.graph, D.graph.nodes, D.graph.edges)
