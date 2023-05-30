from graph_state import GraphState
from collections import Counter
import networkx as nx

# Use 30 qubits
results = []
height = 2
width = 3
g = GraphState(height*width)
dicts = []

for j in range(height):
    for i in range(width-1):
        dicts.append((i+j*width, (i+1)+j*width))

for j in range(height-1):
    for i in range(width):
        dicts.append((i+j*width, i+(j+1)*width))

print(dicts)





for i in dicts:
    g.add_edge(*i)

g.draw()
#N = nx.to_numpy_array(g.to_networkx())
#print(N)

