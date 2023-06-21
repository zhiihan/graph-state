import plotly.graph_objects as go
from grid import Grid
from defect import Defect 
import json
from textwrap import dedent as d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import time
import random
import numpy as np
from helperfunctions import *

height = 4
width = 4
length = 4
shape = [height, length, width]
p = 0.10
seed = 1

G = Grid([height, width, length]) # qubits
D = Defect([height, width, length]) # holes
removed_nodes = G.removed_nodes
log = [] #html version of move_list
move_list = [] #local variable containing moves
camera_state = {
  "scene.camera": {
    "up": {
      "x": 0,
      "y": 0,
      "z": 1
    },
    "center": {
      "x": 0,
      "y": 0,
      "z": 0
    },
    "eye": {
      "x": 1.8999654712209553,
      "y": 1.8999654712209548,
      "z": 1.8999654712209553
    },
    "projection": {
      "type": "perspective"
    }
  }
}

def update_plot(g):
    """
    Main function that updates the plot.
    """
    gnx = g.to_networkx()
    hnx = D.to_networkx()

    for i in removed_nodes:
        gnx.remove_node(i)

    g_nodes, g_edges = nx_to_plot(gnx, shape)
    h_nodes, h_edges = nx_to_plot(hnx, shape)
    x_removed_nodes = [g.node_coords[j][0] for j in removed_nodes]
    y_removed_nodes = [g.node_coords[j][1] for j in removed_nodes]
    z_removed_nodes = [g.node_coords[j][2] for j in removed_nodes]   

    #etext = [f'weight={w}' for w in edge_weights]

    #create a trace for the edges
    trace_edges = go.Scatter3d(
        x=g_edges[0],
        y=g_edges[1],
        z=g_edges[2],
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none')

    #create a trace for the nodes
    trace_nodes = go.Scatter3d(
        x=g_nodes[0],
        y=g_nodes[1],
        z=g_nodes[2],
        mode='markers',
        marker=dict(symbol='circle',
                size=10,
                color='skyblue'),
        )

    trace_removed_nodes = go.Scatter3d(
        x=x_removed_nodes,
        y=y_removed_nodes,
        z=z_removed_nodes,
        mode='markers',
        marker=dict(symbol='circle',
                size=10,
                color='red'),
        visible='legendonly',
        text=[j for j in removed_nodes]
    )

    trace_holes = go.Scatter3d(
        x=h_nodes[0],
        y=h_nodes[1],
        z=h_nodes[2],
        mode='markers',
        marker=dict(symbol='circle',
                size=10,
                color='green'),
        visible='legendonly'
    )

    trace_holes_edges = go.Scatter3d(
        x=h_edges[0],
        y=h_edges[1],
        z=h_edges[2],
        mode='lines',
        line=dict(color='forestgreen', width=2),
        hoverinfo='none',
        visible='legendonly'
    )

    #Include the traces we want to plot and create a figure
    data = [trace_nodes, trace_edges, trace_removed_nodes, trace_holes, trace_holes_edges]
    fig = go.Figure(data=data)
    fig.layout.height = 600
    fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0), scene_camera=camera_state["scene.camera"], legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ))
    return fig

f = update_plot(G)

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
        figure=f
    ),
    dcc.Store(id='draw-plot'),

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
            html.Button('Undo', id='undo'), html.Button('Run Algorithm 1', id='alg1'),
            html.Pre(id='click-data', style=styles['pre'])], className='three columns'),

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
            dcc.Markdown(d("""
                **Select Measurement Basis**

                Click to select the type of measurement.
            """)),
            dcc.RadioItems(['Z', 'Y', 'X'], 'Z', id='radio-items'),
            html.Div(id='slider-output-container'),
            html.Button('Reset Grid', id='reset'), 

            html.Div([html.Button('Reset Seed', id='reset-seed'),
            dcc.Input(id='load-graph-seed', type="text", placeholder="Seed"),]),
            html.Div(
                [dcc.Markdown(d("""
                **Load Graph State**

                Paste data to load a graph state.
                """)),
                dcc.Input(id='load-graph-input', type="text", placeholder="Load Graph State"),
                html.Button('Load Graph', id='load-graph-button'), html.Div(id='loaded')]
           )
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
    Output('draw-plot','data'),
    Input('basic-interactions', 'clickData'), State('radio-items', 'value'), State('click-data', 'children'))
def display_click_data(clickData, measurementChoice, clickLog):
    global removed_nodes
    global move_list
    if not clickData:
        return dash.no_update, dash.no_update
    point = clickData["points"][0]
    # Do something only for a specific trace
    if point["curveNumber"] > 0 or 'x' not in point:
        return dash.no_update, dash.no_update
    else: 
        i = G.get_node_index(point['x'], point['y'], point['z'])
        # Update the plot based on the node clicked
        if i not in removed_nodes:
            removed_nodes.append(i)
            G.handle_measurements(i, measurementChoice)
            move_list.append([i, measurementChoice])
            print('clickedon', i)
    time.sleep(0.1)
    #log.append(f"Move: {len(log)}, Node: {i}, Coordinates: {[point['x'], point['y'], point['z']]}, Measurement: {measurementChoice}")
    log.append(f"{i}, {measurementChoice}; ")
    log.append(html.Br())
    

    return html.P(log), i



@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-interactions', 'relayoutData')],
    State('relayout-data', 'children'))
def display_relayout_data(relayoutData, state):
    global camera_state
    if relayoutData and "scene.camera" in relayoutData:
        camera_state = relayoutData
        return json.dumps(relayoutData, indent=2)
    else:
        return state


@app.callback(
    Output('slider-output-container', 'children'),
    Input('radio-items', 'value'))
def update_output(value):
    return 'You have selected "{}" basis'.format(value)


@app.callback(
    Output('basic-interactions', 'figure', allow_duplicate=True),
    Output('click-data', 'children', allow_duplicate=True),
    Input('reset', 'n_clicks'),
    prevent_initial_call=True)
def reset_grid(input, move_list_reset = True):
    global G
    global D
    global removed_nodes
    global log
    global move_list
    G = Grid([height, width, length])
    D = Defect([height, width, length])
    removed_nodes = []
    fig = update_plot(G)
    log = []
    if move_list_reset:
        move_list = []
    # Make sure the view/angle stays the same when updating the figure        
    return fig, log

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Input('reset-seed', 'n_clicks'),
    State('load-graph-seed', "value"),
    prevent_initial_call=True)
def reset_seed(input, seed):
    """
    Randomly measure qubits.
    """
    fig, log = reset_grid(input)

    if seed is not None:
        random.seed(int(seed))
    # p is the probability of losing a qubit

    measurementChoice = 'Z'

    for i in range(height*length*width):
        if random.random() < p:
            removed_nodes.append(i)
            G.handle_measurements(i, measurementChoice)
            log.append(f"{i}, {measurementChoice}; ")
            log.append(html.Br())
            move_list.append([i, measurementChoice])
            D.add_node(i)
    print(f'Loaded seed : {seed}')
    return log, 1

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('loaded', 'children'),
    Input('load-graph-button', 'n_clicks'),
    State('load-graph-input', "value"),
    prevent_initial_call=True)
def load_graph(n_clicks, input_string):
    reset_grid(n_clicks)

    result = process_string(input_string)

    for i, measurementChoice in result:
        removed_nodes.append(i)
        G.handle_measurements(i, measurementChoice)
        log.append(f"{i}, {measurementChoice}; ")
        log.append(html.Br())
        move_list.append([i, measurementChoice])
    return log, 1, 'Graph loaded!'

def process_string(input_string):
    input_string = input_string.replace(" ", "")
    input_string = input_string[:-1]

    # Split the string into outer lists
    outer_list = input_string.split(";")

    # Split each inner string into individual elements
    result = [inner.split(",") for inner in outer_list]
    for inner in result:
        inner[0] = int(inner[0])
    return result

@app.callback(
    Output('basic-interactions', 'figure', allow_duplicate=True),
    Input('draw-plot', 'data'),
    State('basic-interactions', 'relayoutData'),
    prevent_initial_call=True)
def draw_plot(data, relayoutData):
    """
    Called when ever the plot needs to be drawn.
    """
    fig = update_plot(G)
    # Make sure the view/angle stays the same when updating the figure
    # fig.update_layout(scene_camera=camera_state["scene.camera"])
    return fig

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('loaded', 'children', allow_duplicate=True),
    Input('undo', 'n_clicks'),
    prevent_initial_call=True)
def undo_move(n_clicks):
    if move_list:
        reset_grid(n_clicks, move_list_reset=False)
        
        undo = move_list.pop(-1)
        for move in move_list:
            i, measurementChoice = move
            removed_nodes.append(i)
            G.handle_measurements(i, measurementChoice)
            log.append(f"{i}, {measurementChoice}; ")
            log.append(html.Br())
        return log, 1, f'Undo {undo}'
    else:
        pass

@app.callback(
    Output('click-data', 'children', allow_duplicate=True),
    Output('draw-plot', 'data', allow_duplicate=True),
    Output('loaded', 'children', allow_duplicate=True),
    Input('alg1', 'n_clicks'),
    prevent_initial_call=True)
def algorithm1(nclicks):
    holes = removed_nodes

    hole_locations = np.zeros(4)

    #counting where the holes are
    for h in holes:
        nx, ny, nz = G.node_coords[h]
        for yoffset in range(2):
            for xoffset in range(2):
                if ((nx + xoffset) % 2 == nz % 2) and ((ny + yoffset) % 2 == nz % 2):
                    hole_locations[xoffset+yoffset*2] += 1
    print(hole_locations)
    
    xoffset = np.argmax(hole_locations) // 2
    yoffset = np.argmax(hole_locations) % 2

    print(xoffset, yoffset)


    for z in range(G.shape[2]):
        for y in range(G.shape[1]):
            for x in range(G.shape[0]):
                if ((x + xoffset) % 2 == z % 2) and ((y + yoffset) % 2 == z % 2):
                    i = G.get_node_index(x, y, z)
                    if i not in removed_nodes:
                        G.handle_measurements(i, 'Z')
                        log.append(f"{i}, Z; ")
                        log.append(html.Br())
                        removed_nodes.append(i)
    
    return log, 1, 'Ran Algorithm 1'



app.run_server(debug=True, use_reloader=False)