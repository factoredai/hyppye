# This is to load data
# the graph needs to be prepared; for example utils.data_prep preprocesses and saves prepared edge lists
import networkx as nx

# def load_graph(file_name, directed=False):
#     container = nx.DiGraph() if directed else nx.Graph()
#     G  = nx.read_edgelist(file_name, data=(('weight',float),), create_using=container)
#     G_comp = nx.convert_node_labels_to_integers(G)
#     return G_comp

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

"""
def distortion_row(H1, H2, n, row):
    mc, me, avg, good = 0,0,0,0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):
            (_avg,me,mc) = distortion_entry(H1[i], H2[i],me,mc)
            good        += 1
            avg         += _avg
    avg /= good if good > 0 else 1.0
    return (mc, me, avg, n-1-good)
"""    
