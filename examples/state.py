import numpy as np
import networkx as nx
from helperfunctions import *
from grid import Grid
from holes import Holes


class BrowserState:
    """
    This class contains a local state of the web app, representing the state on the user's local browsing section.
    """

    def __init__(self) -> None:
        self.xmax = 7
        self.ymax = 7
        self.zmax = 7

        self.shape = [self.xmax, self.ymax, self.zmax]

        self.p = 0.09
        self.seed = None
        self.path_clicks = 0

        self.cubes = None
        self.lattice = None
        self.lattice_edges = None
        self.connected_cubes = None

        self.removed_nodes = np.zeros(self.xmax * self.ymax * self.zmax, dtype=bool)
        self.log = []  # html version of move_list
        self.move_list = []  # local variable containing moves
        self.camera_state = {
            "scene.camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 1.8999654712209553, "y": 1.8999654712209548, "z": 1.8999654712209553},
                "projection": {"type": "perspective"},
            }
        }

        self.G = Grid(self.shape)  # qubits
        self.D = Holes(self.shape)  # holes

        self.move_list = []
        self.offset = [None, None, None]
        self.ncubes = None
