import numpy as np
from mpmath import mp

from place_children_codes import *
from place_children import *

# place children. just performs the inversion and then uses the uniform
# unit sphere function to actually get the locations
def add_children_dim(p, x, dim, edge_lengths, use_codes, SB, Gen_matrices, precision):
    mp.prec = precision
    p0, x0  = reflect_at_zero(x, p), reflect_at_zero(x, x)
    #print("Reflection p0: ", p0)
    #print("Reflection x0: ", x0)
    c = len(edge_lengths)
    q = np.linalg.norm(p0)

    # a single child is a special case, place opposite the parent:
    # np.float128(1.0)??????????????????
    #print("P lower: ", p0)
    if c == 1:
        #print("Special case: one child")
        points0 = np.array([mp.mpf(0) for i in range(2*dim)]).reshape((2, dim))
        points0[1, :] = -p0/np.linalg.norm(p0)
    else:
        if use_codes:
            #print("Multiple children case, using codes")
            points0 = place_children_codes(dim, c + 1, True, p0/np.linalg.norm(p0), Gen_matrices)
        else:
            #print("Multiple children case, not using codes")
            points0 = place_children(dim, c + 1, True, p0/np.linalg.norm(p0), True, SB, precision=precision)
        points0 = points0.T

    #print("P upper: ", points0)
    points0[0, :] = p
    #print("Points (before final reflection): ", points0)
    for i in range(1, c + 1):
        #print("$$$$$$$ Params:")
        #print("Edge lengths: ", edge_lengths[i-1, 0])
        #print("Points0: ", points0[i, :])
        points0[i, :] = reflect_at_zero(x, edge_lengths[i-1, 0] * points0[i, :])

    #print("Points (after final reflection): ", points0)
    return points0[1:, :]


# Reflection (circle inversion of x through orthogonal circle centered at a)
def isometric_transform(a, x):
    #print("Iso transform values:")
    #print("a: ", a)
    #print("x: ", x)
    r2 = np.linalg.norm(a)**2 - 1.0
    #print("r2: ", r2)
    return (r2/np.linalg.norm(x - a)**2) * (x - a) + a

# Inversion taking mu to origin
def reflect_at_zero(mu, x):
    a = mu/np.linalg.norm(mu)**2
    #print("Mu/Norm**2: ", a)
    isotrans = isometric_transform(a, x)
    #print("Isotrans: ", isotrans)
    return isotrans