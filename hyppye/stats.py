
import networkx as nx
import numpy as np
import mpmath as mp

###################STILL IN PROGRESSS#################################################################
def entry_is_good(h, h_rec): return (not np.isnan(h_rec)) and (not np.isinf(h_rec)) and h_rec != 0 and h != 0

def distortion_entry(h,h_rec,me,mc):
    avg = abs(h_rec - h)/h
    if h_rec/h > me: me = h_rec/h
    if h/h_rec > mc: mc = h/h_rec
    return (avg,me,mc)

def distortion_row(H1, H2, n, row):
    mc, me, avg, good = 0,0,0,0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):
            (_avg,me,mc) = distortion_entry(H1[i], H2[i],me,mc)
            good        += 1
            avg         += _avg
    avg /= good if good > 0 else 1.0
    return (mc, me, avg, n-1-good)


def map_row(H1, H2, n, row, verbose=False):

    edge_mask = (H1 == 1.0)
    m         = np.sum(edge_mask).astype(int)
    assert m > 0
    if verbose: print(f"\t There are {m} edges for {row} of {n}")
    d = H2
    sorted_dist = np.argsort(d)
    if verbose:
        print(f"\t {sorted_dist[0:5]} vs. {np.array(range(n))[edge_mask]}")
        print(f"\t {d[sorted_dist[0:5]]} vs. {H1[edge_mask]}")
    precs       = np.zeros(m)
    n_correct   = 0
    j = 0
    # skip yourself, you're always the nearest guy
    # TODO (A): j is redundant here
    for i in range(1,n):
        if edge_mask[sorted_dist[i]]:
            n_correct += 1
            precs[j] = n_correct/float(i)
            j += 1
            if j == m:
                break
    return np.sum(precs)/m
