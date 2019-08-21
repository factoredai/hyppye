import numpy as np
import mpmath as mp

def dist(u,v):
    """
    Calculate hyperbolic distance between two points u and v.

    Input:
        * u: Numpy array. Coordinates of a point.
        * v: Numpy array. Coordinates of a point.

    Output:
        * Float. Hyperbolic distance between u and v.
    """

    z  = 2*np.linalg.norm(u-v)**2
    uu = 1 - np.linalg.norm(u)**2
    vv = 1 - np.linalg.norm(v)**2
    return mp.cosh(1+z/(uu*vv))

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
