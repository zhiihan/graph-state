import plotly.graph_objects as go
import networkx as nx

G = nx.Graph()
G.add_edge(1, 2, weight = 1)
G.add_edge(2, 3, weight = 3)
G.add_edge(4, 5, weight = 2)
G.add_edge(6, 3, weight = 4)
G.add_edge(6, 4, weight = 6)

edge_weights =[1,3,2,4,6]

Num_nodes = len(G.nodes)

# plt.figure(figsize=(5,5))
edges = G.edges()

# ## update to 3d dimension
spring_3D = nx.spring_layout(G, dim = 3, k = 0.5) # k regulates the distance between nodes
# weights = [G[u][v]['weight'] for u,v in edges]
# nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold',  width=weights, pos=pos)

# we need to seperate the X,Y,Z coordinates for Plotly
# NOTE: spring_3D is a dictionary where the keys are 1,...,6
x_nodes= [spring_3D[key][0] for key in spring_3D.keys()] # x-coordinates of nodes
y_nodes = [spring_3D[key][1] for key in spring_3D.keys()] # y-coordinates
z_nodes = [spring_3D[key][2] for key in spring_3D.keys()] # z-coordinates

#we need to create lists that contain the starting and ending coordinates of each edge.
x_edges=[]
y_edges=[]
z_edges=[]

#create lists holding midpoints that we will use to anchor text
xtp = []
ytp = []
ztp = []

#need to fill these with all of the coordinates
for edge in edges:
    #format: [beginning,ending,None]
    x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
    x_edges += x_coords
    xtp.append(0.5*(spring_3D[edge[0]][0]+ spring_3D[edge[1]][0]))

    y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
    y_edges += y_coords
    ytp.append(0.5*(spring_3D[edge[0]][1]+ spring_3D[edge[1]][1]))

    z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
    z_edges += z_coords
    ztp.append(0.5*(spring_3D[edge[0]][2]+ spring_3D[edge[1]][2])) 


etext = [f'weight={w}' for w in edge_weights]

trace_weights = go.Scatter3d(x=xtp, y=ytp, z=ztp,
    mode='markers',
    marker =dict(color='rgb(125,125,125)', size=1), #set the same color as for the edge lines
    text = etext, hoverinfo='text')

#create a trace for the edges
trace_edges = go.Scatter3d(
    x=x_edges,
    y=y_edges,
    z=z_edges,
    mode='lines',
    line=dict(color='black', width=2),
    hoverinfo='none')

#create a trace for the nodes
trace_nodes = go.Scatter3d(
    x=x_nodes,
    y=y_nodes,
    z=z_nodes,
    mode='markers',
    marker=dict(symbol='circle',
            size=10,
            color='skyblue')
    )

#Include the traces we want to plot and create a figure
data = [trace_edges, trace_nodes, trace_weights]
fig = go.Figure(data=data)

import dash
from dash import dcc, html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])



app.run_server(debug=True, use_reloader=False)