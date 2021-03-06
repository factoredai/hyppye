import time
import sys
import argparse
import pandas as pd
import networkx as nx
import scipy.sparse.csgraph as csg
from mpmath import mp

from .combinatorial_utils import *
from .data_utils import *
from .math_utils import *
from .evaluation import *
from .screen_utils import welcome_screen

class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))


def setup_parser():
    parser = ArgumentParser(description='parameters for the embeddings')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='path to the .csv file with the data')
    parser.add_argument('-d','--dim', type=int, default = 3,
                        help= 'dimension of the embeding (>2)')
    parser.add_argument('-p','--precision', type=int, default = 256,
                        help='precision desired (in bits)')
    parser.add_argument('-e', '--eps', type=float, default = None,
                        help = 'epsilon for worst-case distortion')
    parser.add_argument('-c','--use_codes', action='store_true',
                        help= 'use coding-theoretic child placement')
    parser.add_argument('-r', '--results',  action='store_true',
                        help= 'print results and metrics of the embedding')
    parser.add_argument('-s', '--sample', type=int,
                        help= 'sample size for computing metrics')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help= 'path for saving the embedding results')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='verbosity level to use, 0 for no verbosity, 1 or more for all verbosity')
    args = parser.parse_args()
    return args

def main():
    welcome_screen()
    time.sleep(2)
    start_time = time.time()
    args = setup_parser()
    verbose = args.verbose > 0
    dataset = args.input
    prec = args.precision
    mp.prec = prec
    if verbose:
        print('Arguments:')
        for arg in vars(args):
            print('{}:'.format(arg), ' '*(18 - len(arg)), getattr(args, arg))

        print("\n=============================")
        print("Combinatorial embedding information:")
        print("Dataset = {}".format(dataset))
        print("Dimensions = {}".format(args.dim))
        print("Precision = {}".format(prec))
        print("Saving embedding to {}".format(args.output))
    
    type_of_file = file_type(args.input)
    
    if type_of_file == 1:
        if verbose:
            print("Received file is an edge list")
        G = load_graph(args.input)
    else:
        if verbose:
            print("Received file is a graph as a csv")
        G = edge_list_build(args.input, args.input)

    
    weighted = is_weighted(G)
    n = G.order()
    num_edges = G.number_of_edges()
    root, d_max = max_degree(G)
    root = 0
    n_bfs = G.order()
    degrees = G.degree()
    path_length  = nx.dag_longest_path_length(G)

    if verbose:
        print("\nGraph information:")
        print("Number of vertices = {}".format(n))
        print("Number of edges = {}".format(num_edges))
        print("Max degree = {}, Max path = {}".format(d_max, path_length))

    if args.eps:    
        if verbose:
            print("Epsilon  = {}".format(args.eps))
        epsilon = args.eps
        tau = get_emb_par(G, 1, epsilon, weighted)
    else:
        r = 1 - mp.eps()/2
        m = mp.log((1+r)/(1-r))
        tau = m/(1.3*path_length)

    if verbose:
        print("Scaling factor tau = ", tau)

    use_codes = False
    if args.use_codes:
        if verbose:
            print("Using coding theoretic child placement")
        use_codes = True
    else:
        if verbose:
            print("Using uniform sphere child placement")

    if args.dim != None and args.dim != 2:
        dim = args.dim
        T = hyp_embedding_dim(G, root, weighted, dim, tau, d_max, use_codes, prec)
    else:
        print("Error: Dimension 2 is not available. Please choose a higher dimension.")
        sys.exit(1)

    end_time = time.time()

    print("Time spent: {} seconds".format(round(end_time - start_time, 2)))

    if args.output != None:
        df = pd.DataFrame(T.astype('float64'))
        df["tau"] = float(tau)
        print("Writing embeddings to disk at path {}".format(args.output))
        df.to_csv(args.output)

    if args.results:
        print("\nComputing quality statistics")
        maps = 0;
        wc = 1;
        d_avg = 0;

        if args.sample != None:
            samples = min(args.sample, n_bfs)
            print("Using {} sample rows for statistics".format(samples))
        else:
            samples = n_bfs

        sample_nodes = np.arange(samples)
        _maps   = np.zeros(samples)
        _d_avgs = np.zeros(samples)
        _wcs    = np.zeros(samples)
        adj_mat_original = nx.to_scipy_sparse_matrix(G,list(range(n_bfs)))
        
        start_time_loop = time.time()
        for i in range(len(sample_nodes)):
                true_dist_row = np.array(csg.dijkstra(adj_mat_original, indices=[sample_nodes[i]], unweighted=(False), directed=False))
                hyp_dist_row = dist_matrix_row(T, sample_nodes[i])/tau
                hyp_dist_row = hyp_dist_row.astype('double')
                n = n_bfs
                curr_map  = map_row(np.squeeze(true_dist_row), np.squeeze(hyp_dist_row[:n]), n, sample_nodes[i])
                _maps[i]  = curr_map
                if verbose:
                    if i % 25 == 0:
                        print("Sample node {}, current MAP = {}".format(sample_nodes[i],curr_map))
                mc, me, avg, bad = distortion_row(np.squeeze(true_dist_row), np.squeeze(hyp_dist_row[:n]),n,sample_nodes[i])
                _wcs[i]  = mc*me
                _d_avgs[i] = avg
                
        end_time_loop = time.time()
        print("Loop time", end_time_loop - start_time_loop)

        maps  = sum(_maps)
        d_avg = sum(_d_avgs)
        wc    = max(_wcs)

        if weighted:
            print("Note: MAP is not well defined for weighted graphs")

        print("Final MAP = {}".format(maps/samples))
        print("Average Distortion = {}\nWorst-Case Distortion = {}".format(d_avg/samples,wc))

if __name__ == '__main__':
    main()
