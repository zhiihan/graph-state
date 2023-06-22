
def get_node_index(x, y, z, shape):
    return x + y * shape[1] + z * shape[1] * shape[2]

def get_node_coords(i, shape):
    index_x = i % shape[0]
    index_y = (i // shape[0]) % shape[1]
    index_z = (i // (shape[0] * shape[1])) % shape[2]
    return [index_x, index_y, index_z]

def nx_to_plot(graph, shape):
    nodes = graph.nodes
    edges = graph.edges
    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: g.node_coords is a dictionary where the keys are 1,...,6

    x_nodes = []
    y_nodes = []
    z_nodes = []

    for j in nodes:
        x_nodes.append(get_node_coords(j, shape)[0]) # x-coordinates of nodes
        y_nodes.append(get_node_coords(j, shape)[1]) # y-coordinates
        z_nodes.append(get_node_coords(j, shape)[2]) # z-coordinate

    #we need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #need to fill these with all of the coordinates
    for edge in edges:
        #format: [beginning,ending,None]
        
        x1 = get_node_coords(edge[0], shape)
        x2 = get_node_coords(edge[1], shape)


        x_coords = [x1[0], x2[0],None]
        x_edges += x_coords

        y_coords = [x1[1], x2[1],None]
        y_edges += y_coords

        z_coords = [x1[2], x2[2],None]
        z_edges += z_coords

    return [x_nodes, y_nodes, z_nodes], [x_edges, y_edges, z_edges]

