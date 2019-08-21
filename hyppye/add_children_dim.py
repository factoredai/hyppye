import numpy as np
from mpmath import mp

from .place_children_codes import *
from .place_children import *

def add_children_dim(p, x, dim, edge_lengths, use_codes, SB, Gen_matrices, precision):
    """
    Algorithm to get the points to embed the children of a node at in the Poincaré Ball.
    Performs the reflections and uniform unit sphere functions.

    Input:
        * p: Numpy array. Coordinates of parent of the node.
        * x: Numpy array. Coordinates of the node whose children will be placed.
        * dim: Integer. Number of dimensions for the embedding.
        * edge_lengths: Numpy array. Length of the edges of the graph.
        * use_codes: Boolean. Whether to use the code-theoretical approach described in the
                     paper or not.
        * SB: Numpy array. Array of coordinates in the embedding.
        * Gen_matrices: Iterable of Numpy arrays. Coding matrices for various dimensions.
        * precision: Integer. Precision (in bits) to use in the encoding.

    Output:
        * Numpy array with the coordinates of the points in the Poincaré ball.
    """
    mp.prec = precision
    p0, x0  = reflect_at_zero(x, p), reflect_at_zero(x, x)

    c = len(edge_lengths)
    q = np.linalg.norm(p0)

    if c == 1:
        points0 = np.array([mp.mpf(0) for i in range(2*dim)]).reshape((2, dim))
        points0[1, :] = -p0/np.linalg.norm(p0)
    else:
        if use_codes:
            points0 = place_children_codes(dim, c + 1, True, p0/np.linalg.norm(p0), Gen_matrices)
        else:
            points0 = place_children(dim, c + 1, True, p0/np.linalg.norm(p0), True, SB, precision=precision)
        points0 = points0.T
    points0[0, :] = p

    for i in range(1, c + 1):
        points0[i, :] = reflect_at_zero(x, edge_lengths[i-1, 0] * points0[i, :])

    return points0[1:, :]


def isometric_transform(a, x):
    """
    Performs the circle inversion of x through an orthogonal circle centered at a.

    Input:
        * a: Numpy array. Coordinates of the reflection center.
        * x: Numpy array. Coordinates of the point the reflect.

    Output:
        * Numpy array with the coordinates of the reflected point.
    """
    r2 = np.linalg.norm(a)**2 - 1.0
    return (r2/np.linalg.norm(x - a)**2) * (x - a) + a


def reflect_at_zero(mu, x):
    """
    Performs the reflection by taking mu to origin.

    Input:
        * mu: Numpy array. Coordinates of point to take to zero.
        * x: Numpy array. Coordinates of point to reflect.

    Output:
        * Numpy array. Coordinates of reflected mu.
    """
    a = mu/np.linalg.norm(mu)**2
    isotrans = isometric_transform(a, x)
    return isotrans
