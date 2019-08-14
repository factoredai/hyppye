import numpy as np

# Express a hyperbolic distance in the unit disk
def hyp_to_euc_dist(x):
    return np.sqrt((np.cosh(x) - np.float128(1))/(np.cosh(x) + np.float128(1)))

def digits(n, pad):
    """
    Equivalent to Julia's digits function, assuming base=2
    """
    bin_repr = list(reversed(np.binary_repr(n)))
    repr_length = len(bin_repr)
    total_length = max(pad, repr_length)
    to_pad = total_length - repr_length
    return np.pad([int(d) for d in bin_repr], (0, to_pad), mode='constant', constant_values=0)
