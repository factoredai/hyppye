# This is to load data
# the graph needs to be prepared; for example utils.data_prep preprocesses and saves prepared edge lists
import networkx as nx

def load_graph(file_name, directed=True):
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
    if len(list(G.edges(data=True))[0][2]):
        return True

    return False


def max_degree(G):
    max_d = 0;
    max_node = -1;

    for deg in G.degree(G.nodes()):
        if deg[1] > max_d:
            max_d = deg[1]
            max_node = deg[0]

    return [max_node, max_d]
