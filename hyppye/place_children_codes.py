import numpy as np
from mpmath import mp

def place_children_codes(dim, n_children, use_sp, sp, Gen_matrices):
    """
    Algorithm to place a set of points on the n-dimensional unit sphere based on coding theory
    The points are placed as the vertices of a hypercube inscribed into the unit sphere, i.e,
    with coordinates (a_1/sqrt(n),...,a_n/sqrt(n)) where n is the dimension and a_i is in {-1,1}.

    It's easy to show that if d = Hamming_distance(a,b), then the Euclidean distance
    between the two vectors is 2sqrt(d/n). We maximize d by using a code.

    In this case, our code is the simplex code,(matrices G) with length 2^z-1, and dimension z
    In this code, the Hamming distance between any pair of vectors is 2^{z-1}.

    Dimension z means that we have 2^z codewords, so we can place up to 2^z children.
    one additional challenge is that our dimensions might be too large, e.g.,
    dim > 2^z-1 for some number of children. Then we generate a codeword and repeat it.

    Note also that the generator matrix for the simplex code is the parity check matrix
    of the Hamming code, which we precompute for all the z's of interest.

    Input:
        * dim: Int. Dimensionality of the embeddings. (2^dim > n_children)
        * n_children: degree of the node.
        * use_sp: Boolean. Whether to use the spherical placing algorithm.
        * sp: Numpy array. Coordinates of the vertices of the hypercube inscribed
              in the hypersphere.
        * Gen_matrices: List of Numpy arrays. List of matrices with the codes to be used.

    Output:
        * Numpy array. Coordinates of points in the n-dimensional unit sphere.
    """

    r = int(np.ceil(np.log2(n_children)))
    n = 2**r - 1
    G = Gen_matrices[r-1]

    C = np.array([mp.mpf(0) for i in range(n_children * dim)]).reshape((n_children, dim))
    for i in range(n_children):
        cw = np.mod(np.dot(np.expand_dims(digits(i, pad=r), -1).T, G), 2)
        rep = int(np.floor(dim/n))
        for j in range(rep):
            C[[i], j*n:(j+1)*n] = cw
        rm = dim - rep*n
        if rm > 0:
            C[[i], rep*n:dim] = cw[0, :rm].T

    points = (1/mp.sqrt(dim)*(-1)**C).T

    if use_sp:
        points = rotate_points(points, sp, dim, n_children)

    return points


def rotate_points(points, sp, N, K):
    """
    Rotates the embedded points in order such that the starting point
    (the parent of the parent node) is the first point embedded.

    Input:
        * points: Numpy array. Coordinates of the points.
        * sp: Numpy array. Coordinates of the parent of the parent node.
        * N: Int. Dimensionality of the embedding.
        * K: Int. number of points.

    Output:
        * Numpy array. Coordinates of rotated points.
    """
    pts = [mp.mpf(0) for i in range(N*K)]
    pts = np.array(pts)
    pts = pts.reshape((N, K))

    x = points[:, 0]
    y = sp.copy()

    u = (x / np.linalg.norm(x))
    v = y - np.dot(np.dot(u.T, y), u)
    v = v / np.linalg.norm(v)
    cost = np.dot(x.T, y)/(np.linalg.norm(x) * np.linalg.norm(y))

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
