import networkx as nx
import numpy as np
from helperfunctions import *
import time


class Holes:
    """
    Create a Grid object as a NetworkX representation.
    """

    def __init__(self, shape, json=None):

        self.shape = shape

        if json:
            self.graph = nx.node_link_graph(json)
        elif shape:
            self.graph = nx.Graph()
        else:
            raise NotImplementedError

        self.big_arrays()

    def add_node(self, i):
        self.graph.add_node(tuple(get_node_coords(i, shape=self.shape)))

    def add_edges(self, double_hole=False):
        nodes = list(self.graph.nodes)
        for index, n in enumerate(nodes):
            for n2 in nodes[index:]:
                if taxicab_metric(n, n2) == 1:
                    self.graph.add_edge(n, n2)
        if double_hole:
            self.double_hole()

    def to_networkx(self):
        self.add_edges()
        return self.graph

    def double_hole(self):
        """
        Check if a hole is a double hole.

        Input: holes object
        Output:
        """

        for i in self.graph.nodes():
            for j in self.graph.nodes():
                x_diff = np.abs(np.array(i) - np.array(j))
                if np.sum(x_diff) == 2:
                    if not ((x_diff[0] == 2) or (x_diff[1] == 2) or (x_diff[2] == 2)):
                        self.graph.add_edge(i, j)

    def find_lattice(self, removed_nodes, xoffset, yoffset, zoffset, max_scale=1):
        """
        Find a raussendorf lattice.

        Returns: cubes: a list containing a cube:

            cube: np.array with shape (19, 3) containing the (x, y, z, scale)
            at [0, :] contains the center of the cube

            n_cubes = the number of cubes found per dimension

        """

        scale = 1
        cubes = []
        centers = [
            np.array([x, y, z], dtype=int)
            for z in range(self.shape[2])
            for y in range(self.shape[1])
            for x in range(self.shape[0])
            if ((x + xoffset) % 2 == (z + zoffset) % 2) and ((y + yoffset) % 2 == (z + zoffset) % 2)
        ]

        n_cubes = np.zeros((self.shape[0] // 2))

        while scale <= max_scale:
            for c in centers:
                for cube_vec in self.cube:
                    arr = c + cube_vec * scale
                    index = get_node_index(*arr, shape=self.shape)
                    # filter out boundary cases
                    if np.any((arr < 0) | np.greater_equal(arr, self.shape)):
                        break
                    # filter out nodes that are measured
                    if removed_nodes[index]:
                        break

                else:
                    cube = np.empty((19, 3), dtype=int)
                    """
                    Format:
                    cube[0, :] = center of the cube
                    cube[:19, :] = coordinates
                    """
                    cube[0, :] = c
                    for i, cube_vec in enumerate(self.cube):
                        cube[i + 1, :3] = c + cube_vec * scale
                        # cube[i,  3] = scale
                    n_cubes[scale - 1] += 1
                    cubes.append(cube)
            scale += 1

        return cubes, n_cubes

    def build_centers_graph(self, cubes):
        """
        Extract the data from the numpy array.

        Returns: the graph of centers C
        """
        C = nx.Graph()  # C is an object that contains all the linked centers

        centers = np.zeros(
            self.shape, dtype=bool
        )  # boolean array that contains whether the node exists or not
        for index, c in enumerate(cubes):
            x, y, z = c[0, :]
            centers[x, y, z] = True
            C.add_node(tuple(c[0, :]))

        edges = []
        for c in cubes:
            for v in self.taxicab2 + self.taxicab3:
                n = c[0, :] + v  # check if distance 2 or 3 node exists
                if centers[n[0], n[1], n[2]]:
                    n1 = tuple(c[0, :])
                    n2 = tuple(n)
                    edges.append((n1, n2))

        C.add_edges_from(edges)
        return C

    def find_max_connected_lattice(self, C):
        """
        Returns the largest subgraph.
        Input: A connected cube: networkx Graph object that is a graph of centers
        Each node is a tuple (x, y, z)
        """
        try:
            largest_cc = max(nx.connected_components(C), key=len)
        except ValueError:
            largest_cc = nx.Graph()
        return largest_cc

    def connected_cube_to_nodes(self, connected_cube):
        """
        Input: A connected cube: networkx Graph object that is a graph of centers
        Each node is a tuple (x, y, z)
        """
        X = nx.Graph()  # X is the same object as C but it contains the actual verticies.

        for node in connected_cube.nodes():
            for cube_vec in self.cube:
                X.add_node(tuple(node + cube_vec))

        nodes = list(X.nodes)
        for index, n in enumerate(nodes):
            for n2 in nodes[index:]:
                if taxicab_metric(n, n2) == 1:
                    X.add_edge(n, n2)
        return X

    def find_connected_lattice(self, C):
        """
        Returns the largest subgraph.
        Input: Graph of centers C
        C contains tuples of coordinates (x, y, z) which is the center of a unit cell.
        """

        connected_cubes = [C.subgraph(c).copy() for c in nx.connected_components(C)]
        return connected_cubes

    def repair_grid(self, p):
        """
        Naive algorithm.
        """
        D = self.graph
        connected_holes = [D.subgraph(d).copy() for d in nx.connected_components(D)]

        weakmeasures = set()
        for subgraph in connected_holes:  # Loop over all subgraphs
            for n in subgraph.nodes:  # Loop over all nodes in a subgraph
                for (
                    boxvec
                ) in self.box:  # For each node, find the vector up/down/left/right/front/back
                    arr = np.array(n) + boxvec
                    if np.any((arr < 0) | np.greater_equal(arr, self.shape)):
                        continue
                    # If this vector is not in the list of weak measurements, add it
                    if tuple(arr) not in weakmeasures:
                        weakmeasures.add(tuple(arr))

        repairs, failures = self.repair(weakmeasures, p)
        return repairs, failures

    def repair(self, weakmeasures, p):
        """
        Given some weak measurements, roll p.
        """
        D = nx.Graph()

        failures = []
        repairs = []
        for i in weakmeasures:
            if np.random.uniform() < p:
                D.add_node(i)
                failures.append(i)
            else:
                repairs.append(i)
        self.graph = D
        return repairs, failures

    def big_arrays(self):
        self.taxicab2 = [
            np.array([-2, 0, 0], dtype=int),
            np.array([-1, -1, 0], dtype=int),
            np.array([-1, 0, -1], dtype=int),
            np.array([-1, 0, 1], dtype=int),
            np.array([-1, 1, 0], dtype=int),
            np.array([0, -2, 0], dtype=int),
            np.array([0, -1, -1], dtype=int),
            np.array([0, -1, 1], dtype=int),
            np.array([0, 0, -2], dtype=int),
            np.array([0, 1, -1], dtype=int),
            np.array([0, 1, 1], dtype=int),
            np.array([1, -1, 0], dtype=int),
            np.array([1, 0, -1], dtype=int),
            np.array([1, 0, 1], dtype=int),
            np.array([1, 1, 0], dtype=int),
        ]

        self.taxicab3 = [
            np.array([-2, -1, 0], dtype=int),
            np.array([-2, 0, -1], dtype=int),
            np.array([-2, 0, 1], dtype=int),
            np.array([-2, 1, 0], dtype=int),
            np.array([-1, -2, 0], dtype=int),
            np.array([-1, -1, -1], dtype=int),
            np.array([-1, -1, 1], dtype=int),
            np.array([-1, 0, -2], dtype=int),
            np.array([-1, 1, -1], dtype=int),
            np.array([-1, 1, 1], dtype=int),
            np.array([0, -2, -1], dtype=int),
            np.array([0, -2, 1], dtype=int),
            np.array([0, -1, -2], dtype=int),
            np.array([0, 1, -2], dtype=int),
            np.array([1, -2, 0], dtype=int),
            np.array([1, -1, -1], dtype=int),
            np.array([1, -1, 1], dtype=int),
            np.array([1, 0, -2], dtype=int),
            np.array([1, 1, -1], dtype=int),
            np.array([1, 1, 1], dtype=int),
        ]

        self.cube = [
            np.array([0, -1, -1]),
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
            np.array([0, 1, 1]),
        ]

        self.box = [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
        ]

    def encode(self):
        return nx.node_link_data(self.graph)
