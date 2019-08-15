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


#######
#This function computes the tau for a required precision i.e:
#if you want a WC_distortion of at most 1 + epsilon this function
# computes the appropiate scaling fcator tau

def get_emb_par(G, k, eps, weighted):
    """
    parameters:
    @G: [netoworx object] a tree
    @k: [int] ???????
    @eps: [Float64] The epsilon that state the required precision
    @weighted: [Boolean] If G is a weighted tree
    """
    n       = G.order()
    degrees = G.degree()
    d_max   = max([cd[1] for cd in dict(degrees).items()])

    (nu, tau) = (0, 0)


    beta    = mp.pi/(1.2*d_max)
    v       = -2*k*mp.log(mp.tan(beta/2))
    m       = length(G.edges())


    if weighted:
        # minimum weight edge:
        w = float('Inf')
        for edge in G.edges()(data=True):
            ew = edge[3]["weight"]
            w  = ew if ew < w  else w

        if w == float('Inf'):
            w = 1
    else:
        w = 1

    _, d_max     = gu.max_degree(G)
    alpha        = 2*mp.pi/(d_max)-2*beta
    _len_        = -2*k*mp.log(mp.tan(alpha/2))
    nu           = _len_/w if (_len_/w > nu) else nu
    tau          = ((1+eps)/eps*v)/w if (1+eps)/eps*v > w*nu else nu

    return tau


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
