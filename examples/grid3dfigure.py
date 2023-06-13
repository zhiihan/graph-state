import plotly.graph_objects as go
import networkx as nx
from grid import Grid
import json
from textwrap import dedent as d
import numpy as np

height = 2
width = 2
length = 2

g = Grid([height, width, length])
gnx = g.to_networkx()


# plt.figure(figsize=(5,5))
edges = gnx.edges()

# ## update to 3d dimension
# g.node_coords = nx.spring_layout(gnx, dim = 3, k = 0.5) # k regulates the distance between nodes
# weights = [G[u][v]['weight'] for u,v in edges]
# nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold',  width=weights, pos=pos)

# we need to seperate the X,Y,Z coordinates for Plotly
# NOTE: g.node_coords is a dictionary where the keys are 1,...,6

x_nodes = [g.node_coords[key][0] for key in g.node_coords.keys()] # x-coordinates of nodes
y_nodes = [g.node_coords[key][1] for key in g.node_coords.keys()] # y-coordinates
z_nodes = [g.node_coords[key][2] for key in g.node_coords.keys()] # z-coordinates

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
    x_coords = [g.node_coords[edge[0]][0],g.node_coords[edge[1]][0],None]
    x_edges += x_coords
    xtp.append(0.5*(g.node_coords[edge[0]][0]+ g.node_coords[edge[1]][0]))

    y_coords = [g.node_coords[edge[0]][1],g.node_coords[edge[1]][1],None]
    y_edges += y_coords
    ytp.append(0.5*(g.node_coords[edge[0]][1]+ g.node_coords[edge[1]][1]))

    z_coords = [g.node_coords[edge[0]][2],g.node_coords[edge[1]][2],None]
    z_edges += z_coords
    ztp.append(0.5*(g.node_coords[edge[0]][2]+ g.node_coords[edge[1]][2])) 

#etext = [f'weight={w}' for w in edge_weights]

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
            color='skyblue'),
    text=[str(i) for i in range(length*width*height)]
    )

#Include the traces we want to plot and create a figure
data = [trace_nodes, trace_edges]
fig = go.Figure(data=data)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=fig
    ),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d("""
                **Hover Data**

                Mouse over values in the graph.
            """)),
            html.Pre(id='hover-data', style=styles['pre'])
        ], className='three columns'),

        html.Div([
            dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown(d("""
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """)),
            html.Pre(id='relayout-data', style=styles['pre']),
        ], className='three columns')
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-interactions', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    [Output('click-data', 'children'),
     Output('basic-interactions', 'figure')],
    [Input('basic-interactions', 'clickData')],
    [State('basic-interactions', 'relayoutData')])
def display_click_data(clickData, relayoutData):
    if not clickData:
        return dash.no_update, dash.no_update
    point = clickData["points"][0]
    print(clickData)
    # Do something only for a specific trace
    if point["curveNumber"] > 0:
        return dash.no_update, dash.no_update
    else: 
        sizes = 8 * np.ones(10)
        sizes[point["pointNumber"]] = 15
        colors = ['blue',]*10
        colors[point["pointNumber"]] = 'red'
        fig.update_traces(marker_size=sizes, marker_color=colors)
    # Make sure the view/angle stays the same when updating the figure
    if relayoutData and "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=relayoutData["scene.camera"])
    return json.dumps(clickData, indent=2), fig


@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-interactions', 'relayoutData')])
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)



app.run_server(debug=True, use_reloader=False)