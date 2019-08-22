import sys
import numpy as np
from mpmath import mp
from .math_utils import digits, coord_from_angle, max_degree, hyp_to_euc_dist

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


def place_children(dim, c, use_sp, sp, sample_from, sb, precision):
    """
    Algorithm to place a set of points uniformly on the n-dimensional unit
    sphere. The idea is to try to tile the surface of the sphere with
    hypercubes. What was done was to build it once with a large number of
    points and then sample for this set of points.

    Input:
        * dim: Integer. Embedding dimensions
        * c: Integer. Number of children
        * use_sp: Boolean. Condition to rotate points or not.
        * sp: the grandparent node
        * sample_from: Boolean. Condition to sample from sb
        * sb: array of coordinates in the embedding
        * precision: precision used in mpmath library

    Output:
        * points: array of coordinates on the hypershpere for each child node
    """

    mp.prec = precision
    N = dim
    K = c

    if sample_from:
        _, K_large = sb.shape

        points = [mp.mpf(0) for i in range(0, N * K)]
        points = np.array(points)
        points = points.reshape((N, K))

        for i in range(0, K - 2 + 1):
            points[:, i] = sb[:, int(np.floor(K_large/(K - 1))) * i]

        points[:, K-1] = sb[:, K_large-1]

        min_d_ds = 2
        for i in range(0, K):
            for j in range(i + 1, K):
                dist = np.linalg.norm(points[:, i] - points[:, j])
                if dist < min_d_ds:
                    min_d_ds = dist

    else:
        if N % 2 == 1:
            AN = N*(2**N) * mp.power(mp.mpf(np.pi), (N - 1) / 2) * mp.mpf(mp.fac((N - 1) // 2) / (mp.fac(N)))
        else:
            AN = N * mp.power(mp.mpf(np.pi), mp.mpf(N / 2)) / (mp.fac((N // 2)) )

        delta = mp.power(mp.mpf(AN/K), ( mp.mpf(1 / (N - 1)) ))

        true_k = 0
        while true_k < K:
            points, true_k = place_on_sphere(delta, N, K, False, precision)
            delta = delta * mp.power(mp.mpf(true_k / K), mp.mpf(1 / (N - 1)))

        points, true_k = place_on_sphere(delta, N, K, True, precision)

        if use_sp:
            points = rotate_points(points, sp, N, K)

    return np.array(points)


def place_on_sphere(delta, N, K, actually_place, precision=100):
    """
    Iterative procedure to get a set of points nearly uniformly on
    the unit hypersphere.

    Input:
        * delta: Float. approximate edge length for N-1 dimensional hypercube
        * N: Integer. Embedding dimension
        * K: Integer. Number or children.
        * actually_place: Boolean. Condition to generate coordinates from
                          angles
        * precision: precision used in mpmath library

    Output:
        * points: array of coordinates on the hypershpere for each child node
        * true_k: number of children nodes
    """
    mp.prec = precision
    mp_sin = np.vectorize(mp.sin)

    points = [mp.mpf(0) for i in range(0, N*K)]
    points = np.array(points)
    points = points.reshape((N, K))

    points_idx = 0
    idx = 0

    curr_angle = [mp.mpf(0) for i in range(0, N-1)]
    curr_angle = np.array(curr_angle)
    curr_angle = curr_angle.reshape((N - 1, 1))

    generate_new_delta = True
    delt_idx = mp.mpf(delta)

    while idx < N-1 and points_idx < K:
        if generate_new_delta:
            if idx == 0:
                delt_idx = mp.mpf(delta)
            else:
                delt_idx = mp.mpf(delta) / mp.mpf( np.prod(mp_sin(curr_angle[0:idx, 0])) )
        if (idx < N-2 and curr_angle[idx, 0] + delt_idx > mp.mpf(np.pi)) or (idx == N-2 and curr_angle[idx, 0] + delt_idx > 2*mp.mpf(np.pi)):
            if idx == 0:
                break
            else:
                generate_new_delta = True
                idx = idx-1
                curr_angle[idx + 1, 0] = mp.mpf(0)
        else:
            curr_angle[idx, 0] = curr_angle[idx, 0] + delt_idx
            if idx == N-2:
                generate_new_delta = False
                if actually_place:
                    point = coord_from_angle(curr_angle, N, precision)
                    points[:, points_idx] = point.flatten()
                points_idx = points_idx+1
            else:
                idx = idx+1
    true_k = points_idx
    return [points, true_k]


def get_emb_par(G, k, eps, weighted):
    """
    Compute the tau for a required precision, i.e:
    If you want a worst-case distortion of at most (1 + epsilon) this function
    computes the appropiate scaling factor tau.

    Input:
        * G: NetworkX DiGraph object. The graph to compute the tau for.
        * k: Int.
        * eps: Float. Epsilon that states the required precision.
        * weighted: Boolean. Whether or not G is a weighted tree.

    Output:
        * Float. The scaling factor tau.
    """
    n       = G.order()
    degrees = G.degree()
    d_max   = max([cd[1] for cd in dict(degrees).items()])

    (nu, tau) = (0, 0)

    beta    = mp.pi/(1.2*d_max)
    v       = -2*k*mp.log(mp.tan(beta/2))
    m       = len(G.edges())

    if weighted:
        w = float('Inf')
        for edge in G.edges()(data=True):
            ew = edge[3]["weight"]
            w  = ew if ew < w  else w
        if w == float('Inf'):
            w = 1
    else:
        w = 1

    _, d_max     = max_degree(G)
    alpha        = 2*mp.pi/(d_max)-2*beta
    _len_        = -2*k*mp.log(mp.tan(alpha/2))
    nu           = _len_/w if (_len_/w > nu) else nu
    tau          = ((1+eps)/eps*v)/w if (1+eps)/eps*v > w*nu else nu

    return tau


def hyp_embedding_dim(G_BFS, root, weighted, dim, tau, d_max, use_codes, precision=100):
    """
    Performs either the code theoretical or the uniform r -dimensional high dimensional embedding.

    Input:
        * G_BFS: NetworkX DiGraph object. The tree to be embedded.
        * root: Int. The root node.
        * weighted: Boolean. Whether or not G is a weighted tree.
        * dim: Int. Dimensionality of the embeddings.
        * tau: Float. Scaling factor.
        * d_max: Int. Maximum degree in the entire tree.
        * use_codes: Boolean. Whether or not to use the code-theoretical approach.
        * precision: Int. Precision (in bits) to use in the embedding.

    Output:
        * Numpy array. Components of the hyperbolic embeddings for each node in the graph.
    """
    n = G_BFS.order()
    mp.prec = precision
    T = np.array([mp.mpf(0) for i in range(n * dim)]).reshape((n, dim))
    root_children = list(G_BFS.successors(root))
    d = len(root_children)
    edge_lengths = hyp_to_euc_dist(tau * np.ones((d, 1)))

    if weighted:
        raise NotImplementedError("Only implemented for unweighted trees currently")

    v = int(np.ceil(np.log2(d_max)))

    if use_codes:
        Gen_matrices = [None]
        for i in range(2, v + 1):
            n = 2**i - 1
            H = np.zeros((i, n))
            for j in range(1, 2**i):
                h_col = digits(j, i)
                H[:, j-1] = h_col
            Gen_matrices.append(H.copy())

    if not use_codes or d_max > dim:
        SB_points = 1000
        SB = place_children(dim, c=SB_points, use_sp=False, sp=0, sample_from=False, sb=0, precision=precision)

    if use_codes and d <= dim:
        R = place_children_codes(dim, n_children=d, use_sp=False, sp=0, Gen_matrices=Gen_matrices)
    else:
        R = place_children(dim, c=d, use_sp=False, sp=0, sample_from=True, sb=SB, precision=precision)

    R = R.T

    for i in range(d):
        R[i, :] *= edge_lengths[i, 0]
        T[root_children[i], :] = R[i, :].copy()

    q = []
    q.extend(root_children)
    node_idx = 0

    while len(q) > 0:
        h = q.pop(0)
        node_idx += 1
        children = list(G_BFS.successors(h))
        parent = list(G_BFS.predecessors(h))
        num_children = len(children)

        if weighted:
            raise NotImplementedError("Weighted graphs not implemented yet.")

        if num_children > 0:
            edge_lengths = hyp_to_euc_dist(tau * np.ones((num_children, 1)))
            q.extend(children)
            if use_codes and num_children + 1 <= dim:
                R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, True, 0, Gen_matrices, precision=precision)
            else:
                R = add_children_dim(T[parent[0], :], T[h, :], dim, edge_lengths, False, SB, 0, precision=precision)

            for i in range(num_children):
                T[children[i], :] = R[i, :]
    return T
