import numpy as np
from mpmath import mp

from place_children_codes import *
from place_children import *

#
def add_children_dim(p, x, dim, edge_lengths, use_codes, SB, Gen_matrices, precision):
    """
     place children. just performs the inversion and then uses the uniform
     unit sphere function to actually get the locations.
     params:
     @ p = parent node
     @ use_codes = a boolean: uses the code theoretical approach or not
     @ SB = array of coordinates in the embedding
     @ Gen_matrices = the coding matrices of each code
     @ precision = precision to be used
     @ x = node whose children are being placed
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
        Reflection (circle inversion of x through orthogonal circle centered at a)
    """
    r2 = np.linalg.norm(a)**2 - 1.0
    return (r2/np.linalg.norm(x - a)**2) * (x - a) + a

def reflect_at_zero(mu, x):
    """
        Inversion taking mu to origin

    """
    a = mu/np.linalg.norm(mu)**2
    isotrans = isometric_transform(a, x)
    return isotrans
