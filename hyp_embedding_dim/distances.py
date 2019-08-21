### distance between rows ####
import numpy as np
import mpmath as mp



def dist(u,v):
    z  = 2*np.linalg.norm(u-v)**2
    uu = 1 - np.linalg.norm(u)**2
    vv = 1 - np.linalg.norm(v)**2
    #print("z", z)
    #print("uu", uu)
    #print("vv", vv)
    #print("mp.cosh(1+z/(uu*vv))", mp.cosh(1+z/(uu*vv)))
    x = 1+z/(uu*vv)
    return mp.log(x + mp.sqrt( x**2 - 1 ) )

# Compute distances from i to all others
def dist_matrix_row(T,i):
    n,_= T.shape
    D = [mp.mpf(0) for x in range(n)]
    D = np.array(D).reshape((1,n))
    for j in range(n):
        D[0,j] = dist(T[i], T[j])

    return D
