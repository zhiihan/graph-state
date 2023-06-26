from grid import Grid

height = 5
width = 5
length = 5

#Algorithms
class Chessboard:
    for x in range(G.shape[0]):
        for y in range(G.shape[1]):
            for z in range(G.shape[2]):
                if (x + y + z) % 2 == 0:
                    i = G.get_node_index(x, y, z)
                    G.handle_measurements(i, basis='Z')
    return G
