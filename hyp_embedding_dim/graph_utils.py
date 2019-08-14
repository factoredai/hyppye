import numpy as np

# Express a hyperbolic distance in the unit disk
def hyp_to_euc_dist(x):
    return np.sqrt((np.cosh(x) - np.float128(1))/(np.cosh(x) + np.float128(1)))

def digits(n, pad):
    """
    Equivalent to Julia's digits function, assuming base=2
    """
    bin_repr = list(reversed(np.binary_repr(n)))
    repr_length = len(bin_repr)
    total_length = max(pad, repr_length)
    to_pad = total_length - repr_length
    return np.pad([int(d) for d in bin_repr], (0, to_pad), mode='constant', constant_values=0)

def max_degree(G):
    max_d = 0;
    max_node = -1;

    for deg in G.degree(G.nodes()):
        if deg[1] > max_d:
            max_d = deg[1]
            max_node = deg[0]

    return [max_node, max_d]

def place_children_codes(dim, n_children, Gen_matrices):
    r = int(np.ceil(np.log2(n_children)))
    n = 2**r - 1

    G = Gen_matrices[r-1]

    #generate the codewords using the matrix
    C = np.zeros((n_children, dim), dtype=np.float128)
    for i in range(n_children):
        #codeword generated from matrix G
        cw = np.mod(np.dot(np.expand_dims(digits(i, pad=r), -1).T, G), 2)

        rep = int(np.floor(dim/n))
        for j in range(rep):
            #repeat it as many times as we can
            C[[i], j*n:(j+1)*n] = cw.astype(np.float128)

        rm = dim - rep*n
        if rm > 0:
            C[[i], rep*n:dim] = cw[0, :rm].T.astype(np.float128)

    # inscribe the unit hypercube vertices into unit hypersphere
    points = (np.float128(1)/np.sqrt(dim)*(-1)**C).T
    return points
