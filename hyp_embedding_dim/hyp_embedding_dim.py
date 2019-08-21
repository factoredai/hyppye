import sys
import numpy as np
import networkx as nx

from add_children_dim import *
from place_children import *
from place_children_codes import *



def hyp_embedding_dim(G_BFS, root, weighted, dim, tau, d_max, use_codes, precision=100):
    """
    This function perform either the code theoretical or the
    uniform r -dimensional high dimensional embedding.
    @G_BFS = the tree to be embbeded
    @ root = the root node
    @ weighted = Boolean
    @ dim = embedding dimensional
    @tau = scaling factor
    @d_max = max degree in the entire tree
    @use_codes = Boolean, use or not the code theoretical approach
    """
    n = G_BFS.order() #returns the number of nodes in the graph
    mp.prec = precision
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
        SB_points = 1000
        SB = place_children(dim, c=SB_points, use_sp=False, sp=0, sample_from=False, sb=0, precision=precision)



    if use_codes and d <= dim:
        R = place_children_codes(dim, n_children=d, use_sp=False, sp=0, Gen_matrices=Gen_matrices)
    else:
        R = place_children(dim, c=d, use_sp=False, sp=0, sample_from=True, sb=SB, precision=precision)

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

        node_idx += 1
        if node_idx % 100 == 0:
            print("Placing children of node {}".format(node_idx))
        children = list(G_BFS.successors(h))
        parent = list(G_BFS.predecessors(h))
        num_children = len(children)


        if weighted:
            raise NotImplementedError("Weighted graphs not implemented yet.")

        if num_children > 0:
            edge_lengths = hyp_to_euc_dist(tau * np.ones((num_children, 1)))

            q.extend(children)

            if use_codes and num_children + 1 <= dim:

                R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, True, 0, Gen_matrices, precision=precision)
            else:

                R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, False, SB, 0, precision=precision)

            for i in range(num_children):

                T[children[i], :] = R[i, :]
    return T



def hyp_to_euc_dist(x):
    """
        Express a hyperbolic distance in the unit disk
    """
    mp_cosh = np.vectorize(mp.cosh)
    mp_sqrt = np.vectorize(mp.sqrt)
    return mp_sqrt((mp_cosh(x) - 1)/(mp_cosh(x) + 1))