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
T = np.zeros((n, dim), dtype=np.float128) #initialize embeddings as zero
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
    R = place_children_codes(dim, n_children=d, use_sp=False, sp=0, Gen_matrices=Gen_matrices)
else:
    raise NotImplementedError("Requires implementation of place_children function")

R = R.T

for i in range(d):
    R[i, :] *= edge_lengths[i] #embeddings of the children
    T[root_children[i], :] = R[i, :].copy() #adding these embeddings to the global embedding matrix

# queue containing the nodes whose children we're placing
q = []
q.extend(root_children)
node_idx = 0

while len(q) > 0:
    h = q[0]
    node_idx += 1
    if node idx % 100 == 0:
        print("Placing children of node {}".format(node_idx))
    children = list(G_BFS.successors(h))
    parent = list(G_BFS.predecessors(h))
    num_children = len(children)
    edge_lengths = hyp_to_euc_dist(tau * np.ones((num_children, 1)))
    q.extend(children)

    if weighted:
        raise NotImplementedError("Weighted graphs not implemented yet.")

    if num_children > 0:
        if use_codes and num_children + 1 <= dim:
            R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, True, 0, Gen_matrices)
        else:
            R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, False, SB, 0)

        for i in range(num_children):
            T[children[i], :] = R[i, :]

print(T)
