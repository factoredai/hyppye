import networkx as nx

def load_graph(file_name, directed=True):
    """
    Creates a graph from a file with the list of edges.

    Input:
        * file_name: String. Path of the file with the list of edges.
        * directed: Boolean. Whether or not the graph is directed.

    Output:
        * NetworkX Graph object.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            if len(tokens) > 2:
                w = float(tokens[2])
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u,v)
    return G


def is_weighted(G):
    """
    Returns whether or not a graph G is weighted.

    Input:
        * G: NetworkX Graph object.

    Output:
        * Boolean. True if G is weighted.
    """
    if len(list(G.edges(data=True))[0][2]):
        return True
    return False


def max_degree(G):
    """
    Returns the maximum degree d_max of a graph G.

    Input:
        * G: NetworkX Graph object.

    Output:
        * Tuple (Int, Int) containing the node with the largest degree and the
          maximum degree of the graph.
    """
    max_d = 0;
    max_node = -1;

    for deg in G.degree(G.nodes()):
        if deg[1] > max_d:
            max_d = deg[1]
            max_node = deg[0]

    return [max_node, max_d]
