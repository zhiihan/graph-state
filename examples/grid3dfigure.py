import plotly.graph_objects as go
from grid import Grid
import json
from textwrap import dedent as d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import time
import random

height = 4
width = 4
length = 4
p = 0.24
seed = 1

G = Grid([height, width, length])
removed_nodes = G.removed_nodes
log = []
graph_states = []
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
    gnx = g.to_networkx()

    # plt.figure(figsize=(5,5))
    
    nodes = gnx.nodes()
    edges = gnx.edges()

    for i in removed_nodes:
        gnx.remove_node(i)

    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: g.node_coords is a dictionary where the keys are 1,...,6

    x_nodes = [g.node_coords[j][0] for j in nodes] # x-coordinates of nodes
    y_nodes = [g.node_coords[j][1] for j in nodes] # y-coordinates
    z_nodes = [g.node_coords[j][2] for j in nodes] # z-coordinates

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
        text=[j for j in nodes]
        )

    #Include the traces we want to plot and create a figure
    data = [trace_nodes, trace_edges]
    fig = go.Figure(data=data)
    fig.layout.height = 600
    fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0), scene_camera=camera_state["scene.camera"]
    )
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
def reset_grid(input):
    global G
    global removed_nodes
    global log
    G = Grid([height, width, length])
    removed_nodes = []
    fig = update_plot(G)
    log = []
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



app.run_server(debug=True, use_reloader=False)