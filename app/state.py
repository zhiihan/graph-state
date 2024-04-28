import numpy as np
from app.utils import *


class BrowserState:
    """
    This class contains a local state of the web app, representing the state on the user's local browsing section.
    """

    def __init__(self) -> None:
        self.xmax = 5
        self.ymax = 5
        self.zmax = 5

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
                "eye": {"x": 1.4, "y": 1.4, "z": 1.3},
                "projection": {"type": "perspective"},
            }
        }

        self.offset = [None, None, None]
        self.xoffset, self.yoffset, self.zoffset = self.offset
        self.ncubes = None
