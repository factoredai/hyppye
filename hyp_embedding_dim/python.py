import sys
import numpy as np
import networkx as nx

sys.path.insert(0, "/home/jgutierrez/Lacuna/projects/hyperbolicEmbeddings_team4/hyp_embedding_dim")

from graph_utils import *

#define parameters
use_codes = True #use coding-theoretic children placement
tau = 1.0
root = 0
weighted = False
dim = 100
d_max = 30 #maximum degree of graph
G = nx.DiGraph([(0,1),(0,2),(0,3),
                (1,4),(2,5),(3,6),(3,7),
                (4,8),(5,9),(6,10),(7,11),(7,12)])#NetworkX Digraph
G_BFS = nx.bfs_tree(G, 0)

n = G_BFS.order() #returns the number of nodes in the graph
T = np.zeros((n, dim)) #initialize embeddings as zero
root_children = list(G_BFS.successors(root)) #get children of the root
d = len(root_children) #get number of children of root
edge_lengths = hyp_to_euc_dist(tau * np.ones((d, 1)))
if weighted:
    raise NotImplementedError("Only implemented for unweighted trees currently")

v = int(np.ceil(np.log2(d_max)))

if use_codes:
    #new way: generate a bunch of generator matrices we'll use for our codes
    Gen_matrices = [None]
    for i in range(2, v + 1):
        n = 2**i - 1
        H = np.zeros((i, n))
        for j in range(1, 2**i):
            h_col = digits(j, i)
            H[:, j-1] = h_col
        Gen_matrices.append(H.copy())

if not use_codes or d_max > dim:
    raise NotImplementedError("Requires implementation of place_children function")

#place the children of the root
if use_codes and d <= dim:
    R = place_children_codes(dim, n_children=d, Gen_matrices=Gen_matrices)
else:
    raise NotImplementedError("Requires implementation of place_children function")
