import plotly.graph_objects as go
from grid import Grid
from holes import Holes
from state import BrowserState
import json
from textwrap import dedent as d
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import time
import random
import numpy as np
import networkx as nx
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()

from helperfunctions import *

# Initialize the state of the user's browsing section
s = BrowserState()
G = Grid(s.shape)
D = Holes(s.shape)
f = update_plot(s, G, D)

styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dcc.Graph(id="basic-interactions", figure=f),
        dcc.Store(id="draw-plot"),
        html.Div(
            className="row",
            children=[
                html.Div(
                    [
                        dcc.Markdown(
                            d(
                                """
                **Hover Data**

                Mouse over values in the graph.
            """
                            )
                        ),
                        html.Pre(id="hover-data", style=styles["pre"]),
                        dcc.Markdown(
                            d(
                                """
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """
                            )
                        ),
                        html.Pre(id="relayout-data", style=styles["pre"]),
                    ],
                    className="three columns",
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            d(
                                """
                **Move Log**

                Click on points in the graph.
            """
                            )
                        ),
                        html.Button("Undo", id="undo"),
                        html.Button("RHG Lattice", id="rhg"),
                        html.Button("Find Lattice", id="findlattice"),
                        html.Button("Run Algorithm 2", id="alg2"),
                        html.Button("Repair Lattice", id="repair"),
                        html.Button("Run Alg 3", id="alg3"),
                        html.Pre(id="click-data", style=styles["pre"]),
                    ],
                    className="three columns",
                ),
                html.Div(
                    [
                        html.Div(id="ui"),
                        dcc.Markdown(
                            d(
                                """
        **Select Measurement Basis**

        Click to select the type of measurement. Click points in the graph to apply measurement.
        """
                            )
                        ),
                        dcc.RadioItems(
                            ["Z", "Y", "X", "Z:Hole"], "Z", id="radio-items", inline=True
                        ),
                        dcc.Markdown(
                            d(
                                """
        **Select display options**
        """
                            )
                        ),
                        dcc.Checklist(
                            ["Qubits", "Holes", "Lattice"],
                            ["Qubits", "Holes", "Lattice"],
                            id="plotoptions",
                        ),
                    ],
                    className="three columns",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Markdown(
                                    d(
                                        """
            **Reset Graph State.**

            Choose cube dimensions as well as a seed. If no seed, will use a random seed.
            """
                                    )
                                ),
                                dcc.Slider(
                                    1,
                                    15,
                                    step=1,
                                    value=s.xmax,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    id="xmax",
                                ),
                                dcc.Slider(
                                    1,
                                    15,
                                    step=1,
                                    value=s.ymax,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    id="ymax",
                                ),
                                dcc.Slider(
                                    1,
                                    15,
                                    step=1,
                                    value=s.zmax,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    id="zmax",
                                ),
                                html.Button("Reset Grid", id="reset"),
                            ]
                        ),
                        dcc.Markdown(
                            d(
                                """
            **Damage the Grid.**

            Select a probability p to randomly remove nodes.
            """
                            )
                        ),
                        dcc.Slider(
                            0,
                            0.3,
                            step=0.03,
                            value=s.p,
                            tooltip={"placement": "bottom", "always_visible": True},
                            id="prob",
                        ),
                        html.Div(
                            [
                                html.Button("Damage Grid", id="reset-seed"),
                                dcc.Input(id="load-graph-seed", type="number", placeholder="Seed"),
                            ]
                        ),
                        html.Div(
                            [
                                dcc.Markdown(
                                    d(
                                        """
                **Load Graph State**

                Paste data to load a graph state.
                """
                                    )
                                ),
                                dcc.Input(
                                    id="load-graph-input",
                                    type="text",
                                    placeholder="Load Graph State",
                                ),
                                html.Button("Load Graph", id="load-graph-button"),
                            ]
                        ),
                    ],
                    className="three columns",
                ),
            ],
        ),
        # dcc.Store stores the intermediate value
        dcc.Store(id="browser-data"),
        dcc.Store(id="graph-data"),
        dcc.Store(id="holes-data"),
        html.Div(id="none", children=[], style={"display": "none"}),
    ]
)


@app.callback(
    Output("browser-data", "data"),
    Output("graph-data", "data"),
    Output("holes-data", "data"),
    Input("none", "children"),
)
def initial_call(dummy):
    """
    Initialize the graph in the browser as a JSON object.
    """
    s = BrowserState()
    G = Grid(s.shape)
    D = Holes(s.shape)

    return jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(Output("hover-data", "children"), [Input("basic-interactions", "hoverData")])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output("click-data", "children"),
    Output("draw-plot", "data"),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("basic-interactions", "clickData"),
    State("radio-items", "value"),
    State("click-data", "children"),
    State("basic-interactions", "hoverData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def display_click_data(
    clickData, measurementChoice, clickLog, hoverData, browser_data, graphData, holeData
):
    """
    Updates the browser state if there is a click.
    """
    if not clickData:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    point = clickData["points"][0]

    # Do nothing if clicked on edges
    if point["curveNumber"] > 0 or "x" not in point:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    else:
        s = jsonpickle.decode(browser_data)
        G = Grid(s.shape, json=graphData)
        D = Holes(s.shape, json=holeData)

        i = get_node_index(point["x"], point["y"], point["z"], s.shape)
        # Update the plot based on the node clicked
        if measurementChoice == "Z:Hole":
            D.add_node(i)
            measurementChoice = "Z"  # Handle it as if it was Z measurement
        if s.removed_nodes[i] == False:
            s.removed_nodes[i] = True
            G.handle_measurements(i, measurementChoice)
            s.move_list.append([i, measurementChoice])
            ui = f"Clicked on {i} at {get_node_coords(i, s.shape)}"
        s.log.append(f"{i}, {measurementChoice}; ")
        s.log.append(html.Br())

        # This solves the double click issue
        time.sleep(0.1)
        return html.P(s.log), i, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("relayout-data", "children"),
    Input("basic-interactions", "relayoutData"),
    State("relayout-data", "children"),
    State("browser-data", "data"),
    prevent_initial_call=True,
)
def display_relayout_data(relayoutData, camera, browser_data):
    """
    Updates zoom and camera information.
    """
    if browser_data is not None:
        s = jsonpickle.decode(browser_data)

    if relayoutData and "scene.camera" in relayoutData:
        s.camera_state = relayoutData
        return json.dumps(relayoutData, indent=2)
    else:
        return camera


@app.callback(
    Output("ui", "children", allow_duplicate=True),
    Input("radio-items", "value"),
    prevent_initial_call=True,
)
def update_output(value):
    return 'You have selected "{}" basis'.format(value)


@app.callback(
    Output("draw-plot", "data", allow_duplicate=True),
    Output("click-data", "children", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("reset", "n_clicks"),
    State("xmax", "value"),
    State("ymax", "value"),
    State("zmax", "value"),
    prevent_initial_call=True,
)
def reset_grid(input, xslider, yslider, zslider):
    """
    Reset the grid.
    """
    s = BrowserState()
    G = Grid(s.shape)
    D = Holes(s.shape)
    s.xmax = int(xslider)
    s.ymax = int(yslider)
    s.zmax = int(zslider)
    s.shape = [s.xmax, s.ymax, s.zmax]
    # Make sure the view/angle stays the same when updating the figure
    return (
        1,
        s.log,
        "Created grid of shape {}".format(s.shape),
        jsonpickle.encode(s),
        G.encode(),
        D.encode(),
    )


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("reset-seed", "n_clicks"),
    State("load-graph-seed", "value"),
    State("prob", "value"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    prevent_initial_call=True,
)
def reset_seed(nclicks, seed_input, prob, browser_data, graphData):
    """
    Randomly measure qubits.
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json=graphData)
    s.p = prob
    D = Holes(s.shape)
    if seed_input:
        # The user has inputted a seed
        random.seed(int(seed_input))
        print(f"Loaded seed : {seed_input}, p = {s.p}")
        ui = "Loaded seed : {}, p = {}".format(seed_input, s.p)
    else:
        # Use a random seed.
        random.seed()
        print(f"Loaded seed : {s.seed}, p = {s.p}")
        ui = "Loaded seed : None, p = {}, shape = {}".format(s.p, s.shape)
    # p is the probability of losing a qubit

    measurementChoice = "Z"

    for i in range(s.xmax * s.ymax * s.zmax):
        if random.random() < s.p:
            if s.removed_nodes[i] == False:
                s.removed_nodes[i] = True
                G.handle_measurements(i, measurementChoice)
                s.log.append(f"{i}, {measurementChoice}; ")
                s.log.append(html.Br())
                s.move_list.append([i, measurementChoice])
                D.add_node(i)
    D.add_edges()
    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("load-graph-button", "n_clicks"),
    State("load-graph-input", "value"),
    State("browser-data", "data"),
    prevent_initial_call=True,
)
def load_graph_from_string(n_clicks, input_string, browser_data):
    s = jsonpickle.decode(browser_data)
    reset_grid(n_clicks, s.xmax, s.ymax, s.zmax, browser_data)

    result = process_string(input_string)

    for i, measurementChoice in result:
        s.removed_nodes[i] = True
        G.handle_measurements(i, measurementChoice)
        s.log.append(f"{i}, {measurementChoice}; ")
        s.log.append(html.Br())
        s.move_list.append([i, measurementChoice])
    return s.log, 1, "Graph loaded!", jsonpickle.encode(s), G.encode(), D.encode()


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
    Output("basic-interactions", "figure"),
    Input("draw-plot", "data"),
    Input("plotoptions", "value"),
    State("basic-interactions", "relayoutData"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
)
def draw_plot(draw_plot, plotoptions, relayoutData, browser_data, graphData, holeData):
    """
    Called when ever the plot needs to be drawn.
    """
    if browser_data is None:
        return dash.no_update

    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json=graphData)
    D = Holes(s.shape, json=holeData)

    fig = update_plot(s, G, D, plotoptions=plotoptions)
    # Make sure the view/angle stays the same when updating the figure
    # fig.update_layout(scene_camera=camera_state["scene.camera"])
    return fig


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("undo", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def undo_move(n_clicks, browser_data, graphData, holeData):
    s = jsonpickle.decode(browser_data)

    if s.move_list:
        # Soft reset
        G = Grid(s.shape)
        s.removed_nodes = np.zeros(s.xmax * s.ymax * s.zmax, dtype=bool)
        s.log = []

        undo = s.move_list.pop(-1)
        for move in s.move_list:
            i, measurementChoice = move
            s.removed_nodes[i] = True
            G.handle_measurements(i, measurementChoice)
            s.log.append(f"{i}, {measurementChoice}; ")
            s.log.append(html.Br())
        return s.log, 1, f"Undo {undo}", jsonpickle.encode(s), G.encode(), D.encode()
    else:
        pass


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("rhg", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def algorithm1(nclicks, browser_data, graphData, holeData):
    """
    Create a RHG lattice from a square lattice.
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json=graphData)
    D = Holes(s.shape, json=holeData)

    holes = D.graph.nodes
    hole_locations = np.zeros(8)
    xoffset, yoffset, zoffset = s.offset

    # counting where the holes are
    for h in holes:
        x, y, z = h
        for zoffset in range(2):
            for yoffset in range(2):
                for xoffset in range(2):
                    if ((x + xoffset) % 2 == (z + zoffset) % 2) and (
                        (y + yoffset) % 2 == (z + zoffset) % 2
                    ):
                        hole_locations[xoffset + yoffset * 2 + zoffset * 4] += 1

    print(hole_locations)

    xoffset = np.argmax(hole_locations) % 2
    yoffset = np.argmax(hole_locations) // 2
    zoffset = np.argmax(hole_locations) // 4

    s.offset = [xoffset, yoffset, zoffset]

    print(f"xoffset, yoffset, zoffset = {(xoffset, yoffset, zoffset)}")

    for z in range(s.shape[2]):
        for y in range(s.shape[1]):
            for x in range(s.shape[0]):
                if ((x + xoffset) % 2 == (z + zoffset) % 2) and (
                    (y + yoffset) % 2 == (z + zoffset) % 2
                ):
                    i = get_node_index(x, y, z, s.shape)
                    if s.removed_nodes[i] == False:
                        G.handle_measurements(i, "Z")
                        s.log.append(f"{i}, Z; ")
                        s.log.append(html.Br())
                        s.removed_nodes[i] = True
                        s.move_list.append([i, "Z"])

    s.cubes, s.n_cubes = D.findlattice(s.removed_nodes, xoffset, yoffset, zoffset)
    ui = f"RHG: Created RHG Lattice."

    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("findlattice", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def findlattice(nclicks, browser_data, graphData, holeData):
    """
    Returns:
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json=graphData)
    D = Holes(s.shape, json=holeData)

    try:
        if s.offset[0] == None:
            # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
            ui = "FindLattice: Run RHG Lattice first."
            return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()

        if s.n_cubes is None:
            s.cubes, s.n_cubes = D.findlattice(s.removed_nodes, s.xoffset, s.yoffset, s.zoffset)

        click_number = nclicks % (len(s.cubes))

        if len(s.cubes) > 0:
            C = nx.Graph()
            C.add_node(tuple(s.cubes[click_number][0, :]))

            X = D.connected_cube_to_nodes(C)

            nodes, edges = nx_to_plot(X, shape=s.shape, index=False)

            lattice = go.Scatter3d(
                x=nodes[0],
                y=nodes[1],
                z=nodes[2],
                mode="markers",
                line=dict(color="blue", width=2),
                hoverinfo="none",
            )

            lattice_edges = go.Scatter3d(
                x=edges[0],
                y=edges[1],
                z=edges[2],
                mode="lines",
                line=dict(color="blue", width=2),
                hoverinfo="none",
            )
            ui = f"FindLattice: Displaying {click_number+1}/{len(s.cubes)} unit cells found for p = {s.p}, shape = {s.shape}"

            s.lattice = lattice.to_json()
            s.lattice_edges = lattice_edges.to_json()
    except NameError:
        # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
        ui = "FindLattice: Run RHG Lattice first."
    return (
        s.log,
        1,
        ui,
        jsonpickle.encode(s),
        G.encode(),
        D.encode(),
    )


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("alg2", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def algorithm2(nclicks, browser_data, graphData, holeData):
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json=graphData)
    D = Holes(s.shape, json=holeData)

    try:
        if s.offset[0] == None:
            # cubes, n_cubes is not defined and this is because we didnt compute the offsets.
            ui = "FindLattice: Run algorithm 1 first."
            return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()

        C = D.build_centers_graph(s.cubes)
        connected_cubes = D.findconnectedlattice(C)
        for i in connected_cubes:
            print(i, len(connected_cubes))

        if len(connected_cubes) > 0:
            click_number = nclicks % (len(connected_cubes))
            X = D.connected_cube_to_nodes(connected_cubes[click_number])

            nodes, edges = nx_to_plot(X, shape=s.shape, index=False)

            lattice = go.Scatter3d(
                x=nodes[0],
                y=nodes[1],
                z=nodes[2],
                mode="markers",
                line=dict(color="blue", width=2),
                hoverinfo="none",
            )

            lattice_edges = go.Scatter3d(
                x=edges[0],
                y=edges[1],
                z=edges[2],
                mode="lines",
                line=dict(color="blue", width=2),
                hoverinfo="none",
            )

            s.lattice = lattice.to_json()
            s.lattice_edges = lattice_edges.to_json()
            ui = f"Alg 2: Displaying {click_number+1}/{len(connected_cubes)}, unit cells = {len(connected_cubes[click_number].nodes)}, edges = {len(connected_cubes[click_number].edges)}"
        else:
            ui = f"Alg 2: No cubes found"
    except TypeError:
        ui = "Alg 2: Run RHG Lattice first."
    except NameError:
        ui = "Alg 2: Run RHG Lattice first."
    return s.log, 2, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("alg3", "n_clicks"),
    State("browser-data", "data"),
    State("graph-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def algorithm3(nclicks, browser_data, graphData, holeData):
    """
    Find a path from the top of the grid to the bottom of the grid.
    """
    s = jsonpickle.decode(browser_data)
    G = Grid(s.shape, json=graphData)
    D = Holes(s.shape, json=holeData)

    gnx = G.to_networkx()

    removed_nodes_reshape = s.removed_nodes.reshape((s.xmax, s.ymax, s.zmax))

    zeroplane = removed_nodes_reshape[:, :, 0]
    zmaxplane = removed_nodes_reshape[:, :, s.zmax - 1]

    x = np.argwhere(zeroplane == 0)  # This is the coordinates of all valid node in z = 0
    y = np.argwhere(zmaxplane == 0)  # This is the coordinates of all valid node in z = L

    path = None
    while path is None:
        try:
            i = get_node_index(*x[s.path_clicks % len(x)], 0, s.shape)
            j = get_node_index(*y[s.path_clicks // len(x)], s.zmax - 1, s.shape)
            path = nx.shortest_path(gnx, i, j)
        except nx.exception.NetworkXNoPath:
            ui = "No path."
            print(f"no path, {i}, {j}")
        finally:
            s.path_clicks += 1

    nodes, edges = path_to_plot(path, s.shape)

    lattice = go.Scatter3d(
        x=nodes[0],
        y=nodes[1],
        z=nodes[2],
        mode="markers",
        line=dict(color="blue", width=2),
        hoverinfo="none",
    )

    lattice_edges = go.Scatter3d(
        x=edges[0],
        y=edges[1],
        z=edges[2],
        mode="lines",
        line=dict(color="blue", width=2),
        hoverinfo="none",
    )

    s.lattice = lattice.to_json()
    s.lattice_edges = lattice_edges.to_json()

    ui = f"Found percolation from z = 0 to z = {s.zmax}"
    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


@app.callback(
    Output("click-data", "children", allow_duplicate=True),
    Output("draw-plot", "data", allow_duplicate=True),
    Output("ui", "children", allow_duplicate=True),
    Output("browser-data", "data", allow_duplicate=True),
    Output("graph-data", "data", allow_duplicate=True),
    Output("holes-data", "data", allow_duplicate=True),
    Input("repair", "n_clicks"),
    State("browser-data", "data"),
    State("holes-data", "data"),
    prevent_initial_call=True,
)
def repair_grid(nclicks, browser_data, holeData):
    s = jsonpickle.decode(browser_data)
    D = Holes(s.shape, json=holeData)

    repairs, failures = D.repair_grid(s.p)

    G = Grid(s.shape)
    s.removed_nodes = np.zeros(s.xmax * s.ymax * s.zmax, dtype=bool)
    s.log = []
    for f in failures:
        i = get_node_index(*f, s.shape)
        s.removed_nodes[i] = True
        G.handle_measurements(i, "Z")
        s.log.append(f"{i}, Z; ")
        s.log.append(html.Br())
        s.move_list.append([i, "Z"])

    if len(repairs) + len(failures) > 0:
        rate = len(repairs) / (len(repairs) + len(failures))
        ui = f"Repairs = {len(repairs)}, Failures = {len(failures)} Repair Rate = {rate:.2f}, Holes = {np.sum(s.removed_nodes)}, peff={np.sum(s.removed_nodes)/(s.xmax*s.ymax*s.zmax)}"
    else:
        ui = "All qubits repaired!"
    return s.log, 1, ui, jsonpickle.encode(s), G.encode(), D.encode()


app.run_server(debug=True, use_reloader=False)
