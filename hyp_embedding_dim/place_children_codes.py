import numpy as np
from mpmath import mp

def place_children_codes(dim, n_children, use_sp, sp, Gen_matrices):

    r = int(np.ceil(np.log2(n_children)))
    n = 2**r - 1

    G = Gen_matrices[r-1]

    #generate the codewords using the matrix
    #C = np.zeros((n_children, dim))
    C = np.array([mp.mpf(0) for i in range(n_children * dim)]).reshape((n_children, dim))
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

    S = np.hstack([u, v])

    R = np.eye(len(x)) - np.dot(u, u.T) - np.dot(v, v.T) + (S).dot(M).dot(S.T)

    for i in range(K):
        pts[:, i] = np.dot(R, points[:, i])
    return pts


def digits(n, pad):
    """
    Equivalent to Julia's digits function, assuming base=2
    """
    bin_repr = list(reversed(np.binary_repr(n)))
    repr_length = len(bin_repr)
    total_length = max(pad, repr_length)
    to_pad = total_length - repr_length
    return np.pad([int(d) for d in bin_repr], (0, to_pad), mode='constant',
                  constant_values=0)
