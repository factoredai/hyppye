import numpy as np
import mpmath as mp

def digits(n, pad):
    """
    Equivalent to Julia's digits function, assuming base=2.

    Represents the number n in base 2, padding as needed to reach length pad.

    Input:
        * n: number to represent.
        * pad: length to pad to.

    Output:
        * Numpy array. Representation of n in base 2, with its respective padding.
    """
    bin_repr = list(reversed(np.binary_repr(n)))
    repr_length = len(bin_repr)
    total_length = max(pad, repr_length)
    to_pad = total_length - repr_length
    return np.pad([int(d) for d in bin_repr], (0, to_pad), mode='constant',
                  constant_values=0)


def coord_from_angle(ang, N, precision=100):
    """
    Spherical coodinates: get Euclidean coord. from a set of points

    Input:
        * ang: angles in spherical coordinates
        * N: Integer. Embedding dimension
        * precision: Integer. Precision (in bits) to use for the encoding.

    Output:
        * point: Array with euclidean coordinates.
    """
    mp.prec = precision
    mp_cos = np.vectorize(mp.cos)
    mp_sin = np.vectorize(mp.sin)

    point = [mp.mpf(0) for i in range(0, N)]
    point = np.array(point)
    point = point.reshape((N, 1))

    for i in range(0, N-1):
        if i == 0:
            point[i] = mp_cos(ang[i, 0])
        else:
            point[i] = np.prod(mp_sin(ang[0:i, 0]))
            point[i] = point[i, 0] * mp_cos(ang[i, 0])

    point[N-1] = mp.mpf(np.prod(mp_sin(ang)))
    return point


def dist(u,v):
    """
    Calculate hyperbolic distance between two points u and v.

    Input:
        * u: Numpy array. Coordinates of a point.
        * v: Numpy array. Coordinates of a point.

    Output:
        * Float. Hyperbolic distance between u and v.
    """
    z  = 2 * np.linalg.norm(u - v)**2

    if np.linalg.norm(u) > 1:
        uu = 1 - (1 - mp.eps)**2
    else:
        uu = 1 - np.linalg.norm(u)**2
    if np.linalg.norm(v) > 1:
        vv = 1 - (1 - mp.eps)**2
    else:
        vv = 1 - np.linalg.norm(v)**2
    x = 1 + z/(uu*vv)

    return mp.log(x + mp.sqrt(x**2 - 1 ))


def dist_matrix_row(T,i):
    """
    Computes distances from a point in T indexed by i and all the other points in T.

    T: Numpy array. Coordinates of points.
    i: Integer. Index of point to compute distances to.
    """
    n,_= T.shape
    D = [mp.mpf(0) for x in range(n)]
    D = np.array(D).reshape((1,n))
    for j in range(n):
        D[0,j] = dist(T[i], T[j])
    return D


def is_weighted(G):
    """
    Returns whether or not a graph G is weighted.

    Input:
        * G: NetworkX Graph object.

    Output:
        * Boolean. True if G is weighted.
    """
    if len(list(G.edges(data=True))[0][2]):
        return True
    return False


def max_degree(G):
    """
    Returns the maximum degree d_max of a graph G.

    Input:
        * G: NetworkX Graph object.

    Output:
        * Tuple (Int, Int) containing the node with the largest degree and the
          maximum degree of the graph.
    """
    max_d = 0;
    max_node = -1;

    for deg in G.degree(G.nodes()):
        if deg[1] > max_d:
            max_d = deg[1]
            max_node = deg[0]

    return [max_node, max_d]


def hyp_to_euc_dist(x):
    """
    Transforms a hyperbolic distance to an euclidean one.

    Input:
        * x: Float. Distance in hyperbolic geometry.

    Output:
        * Float. Distance in euclidean geometry.
    """
    mp_cosh = np.vectorize(mp.cosh)
    mp_sqrt = np.vectorize(mp.sqrt)
    return mp_sqrt((mp_cosh(x) - 1)/(mp_cosh(x) + 1))
