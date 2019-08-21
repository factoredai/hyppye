### distance between rows ####
import numpy as np
import mpmath as mp
import pdb


def dist(u,v):
    """
        computes the hyperbolic distance between two vectors. To avoid
        precision problems is a vector has norm biiger than one the norm is
        changed by the maximum posible value (1-eps)
    """


    z  = 2*np.linalg.norm(u-v)**2
    if np.linalg.norm(u) > 1:
        uu =1 -(1 - mp.eps)**2
    else:
        uu = 1 - np.linalg.norm(u)**2
    if np.linalg.norm(v) > 1:
        vv =1 -(1 - mp.eps)**2
    else:
        vv = 1 - np.linalg.norm(v)**2


    x = 1 + z/(uu*vv)



    return mp.log(x + mp.sqrt(x**2 - 1 ))


def dist_matrix_row(T,i):
    """
        Compute distances from i to all others
    """
    n,_= T.shape
    D = [mp.mpf(0) for x in range(n)]
    D = np.array(D).reshape((1,n))
    for j in range(n):
        D[0,j] = dist(T[i], T[j])

    return D
