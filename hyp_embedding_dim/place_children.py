import numpy as np
from mpmath import mp
from place_children_codes import rotate_points


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
                # reset the angle down:
                curr_angle[idx + 1, 0] = mp.mpf(0)

        else:
            curr_angle[idx, 0] = curr_angle[idx, 0] + delt_idx

            if idx == N-2:
                generate_new_delta = False

                # generate point from spherical coordinates
                if actually_place:
                    point = coord_from_angle(curr_angle, N, precision)
                    points[:, points_idx] = point.flatten()

                points_idx = points_idx+1
            else:
                idx = idx+1

    true_k = points_idx
    return [points, true_k]


def coord_from_angle(ang, N, precision=100):
    """
    Spherical coodinates: get Euclidean coord. from a set of points

    Input:
        * ang: angles in spherical coordinates
        * N: Integer. Embedding dimension

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
