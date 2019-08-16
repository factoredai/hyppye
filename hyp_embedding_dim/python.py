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
dim = 3
d_max = 3 #maximum degree of graph
G = nx.DiGraph([(0,1),(0,2),(0,3),(2,4),(3,5),(5,6),(5,7)]) #NetworkX Digraph
G_BFS = nx.bfs_tree(G, 0)

n = G_BFS.order() #returns the number of nodes in the graph
mp.prec = 256
T = np.array([mp.mpf(0) for i in range(n * dim)]).reshape((n, dim))
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
    R[i, :] *= edge_lengths[i, 0] #embeddings of the children
    T[root_children[i], :] = R[i, :].copy() #adding these embeddings to the global embedding matrix

# queue containing the nodes whose children we're placing
q = []
q.extend(root_children)
node_idx = 0

while len(q) > 0:
    h = q.pop(0)
    print('Popped node', h)
    node_idx += 1
    if node_idx % 100 == 0:
        print("Placing children of node {}".format(node_idx))
    children = list(G_BFS.successors(h))
    parent = list(G_BFS.predecessors(h))
    num_children = len(children)
    print('N children:', num_children)

    if weighted:
        raise NotImplementedError("Weighted graphs not implemented yet.")

    if num_children > 0:
        edge_lengths = hyp_to_euc_dist(tau * np.ones((num_children, 1)))
        print('Adding children to queue:', children)
        q.extend(children)

        if use_codes and num_children + 1 <= dim:
            #print('Embedding children using codes')
            #print('p:', T[parent[0], :])
            #print('x:', T[h, :])
            #print('edge lengths', edge_lengths)
            R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, True, 0, Gen_matrices)
        else:
            #print('Embedding children without codes')
            #print('p:', T[parent[0], :])
            #print('x:', T[h, :])
            #print('edge lengths', edge_lengths)
            R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, False, SB, 0)

        for i in range(num_children):
            print('Embedding for children', children[i])
            print(R[i, :])
            T[children[i], :] = R[i, :]
