from graph_state import GraphState
import networkx as nx
from app.utils import *


class Grid(GraphState):
    """
    Create a Grid object from a GraphState object.
    """

    def __init__(self, shape, json=None):
        # Decoding from nx_object:
        if json:
            self.graph = nx.node_link_graph(json)
        elif shape:
            self.shape = shape
            edges = self.generate_cube_edges()
            self.graph = nx.from_edgelist(edges)
        else:
            raise NotImplementedError

        super().__init__(self.graph.order())

        for i in range(self.graph.order()):
            self.h(i)

        for e in self.graph.edges:
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
        return edges

    def adjaencyMatrix(self):
        return nx.to_numpy_array(self.to_networkx())

    def handle_measurements(self, i, basis):
        self.measure(i, basis=basis)

    def encode(self):
        return nx.node_link_data(self.to_networkx())
