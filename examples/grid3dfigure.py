import plotly.graph_objects as go
import networkx as nx
from grid import Grid
import json
from textwrap import dedent as d
import numpy as np

height = 4
width = 4
length = 4

g = Grid([height, width, length])
removed_nodes = []

def update_plot(g):
    gnx = g.to_networkx()

    # plt.figure(figsize=(5,5))
    edges = gnx.edges()

    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: g.node_coords is a dictionary where the keys are 1,...,6

    x_nodes = [g.node_coords[i][0] for i in g.node_coords.keys() if i not in removed_nodes] # x-coordinates of nodes
    y_nodes = [g.node_coords[i][1] for i in g.node_coords.keys() if i not in removed_nodes] # y-coordinates
    z_nodes = [g.node_coords[i][2] for i in g.node_coords.keys() if i not in removed_nodes] # z-coordinates

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
    fig.layout.height = 800
    fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

fig = update_plot(g)

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
        ], className='three columns'),
        html.Div([
            dcc.RadioItems(['Z', 'Y', 'X'], 'Z', id='radio-items'),
            html.Button('Reset Grid', id='reset'),
        html.Div(id='slider-output-container')
        ], className='three columns'),
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-interactions', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    Output('basic-interactions', 'figure'),
    Input('basic-interactions', 'clickData'), Input('radio-items', 'value'),
    State('basic-interactions', 'relayoutData'))

def display_click_data(clickData, measurementChoice, relayoutData):
    global removed_nodes
    if not clickData:
        return dash.no_update, dash.no_update
    point = clickData["points"][0]
    # Do something only for a specific trace
    i = g.get_node_index(point['x'], point['y'], point['z'])
    if point["curveNumber"] > 0:
        return dash.no_update, dash.no_update
    else: 
        # Update the plot based on the node clicked
        print('clickedon', i)
        g.handle_measurements(i, measurementChoice)
        removed_nodes.append(i)
        fig = update_plot(g)

    # Make sure the view/angle stays the same when updating the figure
    if relayoutData and "scene.camera" in relayoutData:
        fig.update_layout(scene_camera=relayoutData["scene.camera"])
    return json.dumps(clickData, indent=2), fig


@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-interactions', 'relayoutData')])
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)

@app.callback(
    Output('slider-output-container', 'children'),
    Input('radio-items', 'value'))
def update_output(value):
    return 'You have selected "{}" basis'.format(value)


@app.callback(
    Output('basic-interactions', 'figure', allow_duplicate=True),
    Input('reset', 'n_clicks'),
    prevent_initial_call=True)
def reset_grid(input):
    print(input, 'hi')
    global g 
    global removed_nodes
    g = Grid([height, width, length])
    removed_nodes = []
    fig = update_plot(g)
    return fig

app.run_server(debug=True, use_reloader=False)