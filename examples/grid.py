from graph_state import GraphState
from collections import Counter
import networkx as nx

# Use 30 qubits
results = []
height = 5
width = 5 #should be same as height for now
dicts = []

class Grid(GraphState):
    def __init__(self, width, height):
        self.height = height
        self.width = width
        super().__init__(width*height)
        self.edges = []
        
    def make_grid(self):
        for j in range(self.height):
            for i in range(self.width-1):
                self.edges.append((i+j*self.width, (i+1)+j*self.width))

        for j in range(self.height-1):
            for i in range(self.width):
                self.edges.append((i+j*width, i+(j+1)*width))

        for e in self.edges:
            self.add_edge(*e)


g = Grid(height, width)
g.make_grid()
print(nx.to_numpy_array(g.to_networkx()))
g.draw()
#N = nx.to_numpy_array(g.to_networkx())
#print(N)

