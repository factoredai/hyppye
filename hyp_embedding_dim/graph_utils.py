import numpy as np
from mpmath import mp

# Express a hyperbolic distance in the unit disk
def hyp_to_euc_dist(x):
    mp_cosh = np.vectorize(mp.cosh)
    mp_sqrt = np.vectorize(mp.sqrt)
    return mp_sqrt((mp_cosh(x) - 1))/(mp_cosh(x) + 1)

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

#  algorithm to place a set of points uniformly on the n-dimensional unit sphere
#  see: http://www02.smt.ufrj.br/~eduardo/papers/ri15.pdf
#  the idea is to try to tile the surface of the sphere with hypercubes
#  works well for larger numbers of points
#  what we'll actually do is to build it once with a large number of points
#  and then sample for this set of points
def place_children(dim, c, use_sp, sp, sample_from, sb):
    raise NotImplementedError()

# rotate the set of points so that the first vector
# coincides with the starting point sp
# N = dimension, K = # of points
def rotate_points(points, sp, N, K):
    pts = [mp.mpf(0) for i in range(N*K)]
    pts = np.array(pts)
    pts = pts.reshape((N, K))

    x = points[:, 0]
    y = sp.copy()

    # rotate x to y
    u = (x / np.linalg.norm(x))
    v = y - np.dot(np.dot(u.T, y), u)
    v = v / np.linalg.norm(v)
    cost = np.dot(x.T, y)/(np.linalg.norm(x) * np.linalg.norm(y))

    #no rotation needed
    if 1.0 - cost**2 <= 0.0:
        return points

    sint = mp.sqrt(1.0 - cost**2)
    u = u.reshape((-1, 1))
    v = v.reshape((-1, 1))

    M = np.array([[cost, -sint], [sint, cost]])
    S = np.vstack([u, v])
    R = np.eye(len(x)) - np.dot(u, u.T) - np.dot(v, v.T) + (S.T).dot(M).dot(S)

    for i in range(K):
        pts[:, i] = np.dot(R, points[:, i])
    return pts


def place_children_codes(dim, n_children, use_sp, sp, Gen_matrices):

    r = int(np.ceil(np.log2(n_children)))
    n = 2**r - 1

    G = Gen_matrices[r-1]

    #generate the codewords using the matrix
    C = np.zeros((n_children, dim))
    for i in range(n_children):
        #codeword generated from matrix G
        cw = np.mod(np.dot(np.expand_dims(digits(i, pad=r), -1).T, G), 2)

        rep = int(np.floor(dim/n))
        for j in range(rep):
            #repeat it as many times as we can
            C[[i], j*n:(j+1)*n] = cw

        rm = dim - rep*n
        if rm > 0:
            C[[i], rep*n:dim] = cw[0, :rm].T

    # inscribe the unit hypercube vertices into unit hypersphere
    points = (1/mp.sqrt(dim)*(-1)**C).T

    # rotate to match the parent, if we need to
    if use_sp:
        points = rotate_points(points, sp, dim, n_children)

    return points

# Reflection (circle inversion of x through orthogonal circle centered at a)
def isometric_transform(a, x):
    r2 = np.linalg.norm(a)**2 - 1.0
    return (r2/np.linalg.norm(x - a)**2) * (x - a) + a

# Inversion taking mu to origin
def reflect_at_zero(mu, x):
    a = mu/np.linalg.norm(mu)**2
    return isometric_transform(a, x)

# place children. just performs the inversion and then uses the uniform
# unit sphere function to actually get the locations
def add_children_dim(p, x, dim, edge_lengths, use_codes, SB, Gen_matrices):
    p0, x0  = reflect_at_zero(x, p), reflect_at_zero(x, x)
    c = len(edge_lengths)
    q = np.linalg.norm(p0)

    # a single child is a special case, place opposite the parent:
    # np.float128(1.0)??????????????????
    if c == 1:
        points0 = np.array([mp.mpf(0) for i in range(2*dim)])
        points0 = np.reshape(points0, (2, dim))
        points0[1, :] = p0/np.linalg.norm(p0)
    else:
        if use_codes:
            points0 = place_children_codes(dim, c + 1, True, p0/np.linalg.norm(p0), Gen_matrices)
        else:
            points0 = place_children(dim, c + 1, True, p0/np.linalg.norm(p0), True, SB)
        points0 = points0.T

    points0[0, :] = p
    for i in range(1, c + 1):
        points0[i, :] = reflect_at_zero(x, edge_lengths[i-1, 0] * points0[i, :])

    return points0[1:, :]
