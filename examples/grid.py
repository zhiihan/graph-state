from graph_state import GraphState
from collections import Counter
import random
import networkx as nx

# Use 30 qubits
results = []
height = 2
width = 2 #should be same as height for now
dicts = []

class Grid(GraphState):
    def __init__(self, width, height):
        self.height = height
        self.width = width
        super().__init__(width*height)
        self.edges = []
        self.removed = []
        
        for j in range(self.height):
            for i in range(self.width-1):
                self.edges.append((i+j*self.width, (i+1)+j*self.width))

        for j in range(self.height-1):
            for i in range(self.width):
                self.edges.append((i+j*self.width, i+(j+1)*self.width))

        for i in range(self.height*self.width):
            self.h(i)
        
        for e in self.edges:
            self.cz(*e)



    def damage_grid(self, p):
        # p is the probability of losing a qubit
        for i in range(self.height*self.width):
            if random.random() < p:
                g.measure(i)
                self.removed.append(i)



g = Grid(height, width)

g.measure(1)


n = g.to_networkx()
for i in g.removed:
    n.remove_node(i)
print(nx.to_numpy_array(n))


print(nx.to_numpy_array(g.to_networkx()))
g.draw()
#N = nx.to_numpy_array(g.to_networkx())
#print(N)

