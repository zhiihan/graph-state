import numpy as np
from graph_state import GraphState
from collections import Counter
import plotly.graph_objects as go


def get_node_index(x, y, z, shape):
    return x + y * shape[0] + z * shape[1] * shape[0]


def get_node_coords(i, shape):
    index_x = i % shape[0]
    index_y = (i // shape[0]) % shape[1]
    index_z = (i // (shape[0] * shape[1])) % shape[2]
    return [index_x, index_y, index_z]


def nx_to_plot(graph, shape, index=True):
    nodes = graph.nodes
    edges = graph.edges
    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: g.node_coords is a dictionary where the keys are 1,...,6

    x_nodes = []
    y_nodes = []
    z_nodes = []

    # if we pass in the index
    for j in nodes:
        if index:
            x, y, z = get_node_coords(j, shape)
        else:
            x = j[0]
            y = j[1]
            z = j[2]
        x_nodes.append(x)  # x-coordinates of nodes
        y_nodes.append(y)  # y-coordinates
        z_nodes.append(z)  # z-coordinate

    # we need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []

    # need to fill these with all of the coordinates
    for edge in edges:
        # format: [beginning,ending,None]
        if index:
            x1 = get_node_coords(edge[0], shape)
            x2 = get_node_coords(edge[1], shape)
        else:
            x1 = edge[0]
            x2 = edge[1]

        x_coords = [x1[0], x2[0], None]
        x_edges += x_coords

        y_coords = [x1[1], x2[1], None]
        y_edges += y_coords

        z_coords = [x1[2], x2[2], None]
        z_edges += z_coords

    return [x_nodes, y_nodes, z_nodes], [x_edges, y_edges, z_edges]


def path_to_plot(path, shape, index=True):
    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: g.node_coords is a dictionary where the keys are 1,...,6

    x_nodes = []
    y_nodes = []
    z_nodes = []

    # if we pass in the index
    for j in path:
        if index:
            x, y, z = get_node_coords(j, shape)
        else:
            x = j[0]
            y = j[1]
            z = j[2]
        x_nodes.append(x)  # x-coordinates of nodes
        y_nodes.append(y)  # y-coordinates
        z_nodes.append(z)  # z-coordinate

    # we need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []

    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # need to fill these with all of the coordinates
    for edge in edges:
        # format: [beginning,ending,None]
        if index:
            x1 = get_node_coords(edge[0], shape)
            x2 = get_node_coords(edge[1], shape)
        else:
            x1 = edge[0]
            x2 = edge[1]

        x_coords = [x1[0], x2[0], None]
        x_edges += x_coords

        y_coords = [x1[1], x2[1], None]
        y_edges += y_coords

        z_coords = [x1[2], x2[2], None]
        z_edges += z_coords

    return [x_nodes, y_nodes, z_nodes], [x_edges, y_edges, z_edges]


def taxicab_metric(node1, node2):
    x1 = np.array(node1)
    x2 = np.array(node2)

    return np.sum(np.abs(x1 - x2))


def update_plot(s, g, d, plotoptions=["Qubits", "Holes", "Lattice"]):
    """
    Main function that updates the plot.
    """

    gnx = g.to_networkx()
    hnx = d.to_networkx()

    for i, value in enumerate(s.removed_nodes):
        if value == True:
            gnx.remove_node(i)

    g_nodes, g_edges = nx_to_plot(gnx, s.shape)
    h_nodes, h_edges = nx_to_plot(hnx, s.shape, index=False)
    # x_removed_nodes = [g.node_coords[j][0] for j in removed_nodes]
    # y_removed_nodes = [g.node_coords[j][1] for j in removed_nodes]
    # z_removed_nodes = [g.node_coords[j][2] for j in removed_nodes]

    # create a trace for the edges
    trace_edges = go.Scatter3d(
        x=g_edges[0],
        y=g_edges[1],
        z=g_edges[2],
        mode="lines",
        line=dict(color="black", width=2),
        hoverinfo="none",
    )

    # create a trace for the nodes
    trace_nodes = go.Scatter3d(
        x=g_nodes[0],
        y=g_nodes[1],
        z=g_nodes[2],
        mode="markers",
        marker=dict(symbol="circle", size=10, color="skyblue"),
    )

    trace_holes = go.Scatter3d(
        x=h_nodes[0],
        y=h_nodes[1],
        z=h_nodes[2],
        mode="markers",
        marker=dict(symbol="circle", size=10, color="green"),
    )

    trace_holes_edges = go.Scatter3d(
        x=h_edges[0],
        y=h_edges[1],
        z=h_edges[2],
        mode="lines",
        line=dict(color="forestgreen", width=2),
        hoverinfo="none",
    )

    if "Qubits" in plotoptions:
        trace_nodes.visible = True
        trace_edges.visible = True
    else:
        trace_nodes.visible = "legendonly"
        trace_edges.visible = "legendonly"

    if "Holes" in plotoptions:
        trace_holes.visible = True
        trace_holes_edges.visible = True
    else:
        trace_holes.visible = "legendonly"
        trace_holes_edges.visible = "legendonly"

    # Include the traces we want to plot and create a figure
    data = [trace_nodes, trace_edges, trace_holes, trace_holes_edges]
    if s.lattice:
        if "Lattice" in plotoptions:
            s.lattice.visible = True
        else:
            s.lattice.visible = "legendonly"
        data.append(s.lattice)
    if s.lattice_edges:
        if "Lattice" in plotoptions:
            s.lattice_edges.visible = True
        else:
            s.lattice_edges.visible = "legendonly"
        data.append(s.lattice_edges)

    fig = go.Figure(data=data)
    fig.layout.height = 600
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene_camera=s.camera_state["scene.camera"],
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig
