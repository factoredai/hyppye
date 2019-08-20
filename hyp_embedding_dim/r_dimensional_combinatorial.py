import scipy.sparse.csgraph as csg
import argparse
import networkx as nx
from mpmath import mp
import time
import pandas as pd
import sys

from distances import *
from utils import ArgumentParser
from embedding_parameter import *
from load_graph import *
from hyp_embedding_dim import *
from stats import *


parser = ArgumentParser(description='parameters for the embeddings')


parser.add_argument('-d', '--dataset', metavar='dataset', type=str,
                    help='dataset to embed', required=True)


parser.add_argument('-l','--dim', metavar = 'dimension', type=int,
                    default = 3, help= 'dimension of the embeding (>2)')


parser.add_argument('-p','--precision', metavar = 'precision', type=int,
                    default = 256, help='precision required')


parser.add_argument('-e', '--eps', metavar = 'epsilon distortion', type=float,
                    default = 0.1, help = 'epsilon for WC_distortion distortion')


parser.add_argument('-a','--auto_tau', action='store_true',
                    help= 'Calculate scale assuming 64-bit final embedding')


parser.add_argument('-c','--code_thoery', action='store_true',
                    help= 'Use coding-theoretic child placement')


parser.add_argument('-s', '--stats',  action='store_true',
                    help= 'Get statistics')


parser.add_argument('-ss', '--stats_sample', metavar = 'sample for computing evaluation metrics',
                    type=int,help= 'Get statistics')


parser.add_argument('--y', action='store_true',
                    help= 'Save the distance matrix')


parser.add_argument('-m', '--save_embedding', metavar='name of saved file', type=str,
                    help= 'Save embedding to file', required=True)


parser.add_argument('--w', metavar = 'visualize',
                    help= 'Visualize the embedding (only for 2 dimensions)')


args = parser.parse_args()
print('args', args)
print("\n=============================")
print("Combinatorial Embedding. Info:")
print("Data set = {}".format(args.dataset))
print("Dimensions = {}".format(args.dim))



prec = args.precision
mp.prec =  prec
print("Precision = {}".format(prec))

if not args.save_embedding:
    print("No file specified to save embedding!")
else:
    print("Save embedding to {}".format(args.save_embedding))

print("mp.eps", mp.eps)

######-------------LOAD THE GRAPH--------------######

G = load_graph(args.dataset)
######-------------LOAD THE GRAPH--------------######

weighted = is_weighted(G)


print("\nGraph information")


n = G.order()
print("Number of vertices = {}".format(n))


num_edges = G.number_of_edges()
print("Number of edges = {}".format(num_edges))

root, d_max   = max_degree(G)
root = 0



# A few statistics
n_bfs   = G.order()
degrees = G.degree()
path_length  = nx.dag_longest_path_length(G)
print("Max degree = {}, Max path = {}".format(d_max, path_length))



start_time = time.time()

if args.auto_tau:
    #naive tau, max posible distance divided by path length
    r = 1 - mp.eps()/2
    m = mp.log((1+r)/(1-r))
    tau = m/(1.3*path_length)
elif args.eps != None:
    print("Epsilon  = {}".format(args.eps))
    epsilon = args.eps
    #tau for a given precision
    tau = get_emb_par(G, 1, epsilon, weighted)
else:

    print("if auto-tau (-a) is not a parameter then a precision must be given")
    sys.exit(1)

print("Scaling factor tau = ", tau)

use_codes = False
if args.code_thoery:
    print("Using coding theoretic child placement")
    use_codes = True
else:
    print("Using uniform sphere child placement")


if args.dim != None and args.dim != 2:
    dim = args.dim
    T = hyp_embedding_dim(G, root, weighted, dim, tau, d_max, use_codes, prec)
else:
    print("Dimension 2 is not available. Please choose a higher dimension")
    sys.exit(1)

end_time = time.time()


print("Time spent: {} seconds".format(end_time - start_time))


if args.save_embedding != None:

    df = pd.DataFrame(T.astype('float64'))
    df["tau"] = float(tau)
    df.to_csv(args.save_embedding)







#####------------------Evaluation------------------------####
if args.stats:
    print("\nComputing quality statistics")
    maps = 0;
    wc = 1;
    d_avg = 0;


    if args.stats_sample != None:
        samples = min(parsed_args.stats_sample, n_bfs)
        print("Using {} sample rows for statistics".format(samples))
    else:
        samples = n_bfs

    sample_nodes = np.random.permutation(n_bfs)[:samples]

    _maps   = np.zeros(samples)
    _d_avgs = np.zeros(samples)
    _wcs    = np.zeros(samples)


    adj_mat_original    =  nx.to_scipy_sparse_matrix(G,list(range(n_bfs-1)))



    for i in range(len(sample_nodes)):

            true_dist_row = np.array(csg.dijkstra(adj_mat_original, indices=[sample_nodes[i]-1], unweighted=(False), directed=False))


            hyp_dist_row = dist_matrix_row(T, sample_nodes[i])/tau


            n = n_bfs

            curr_map  = map_row(true_dist_row.T, hyp_dist_row[:n].T, n-1, sample_nodes[i]-1)
            _maps[i]  = curr_map


            print("Row {}, current MAP = {}".format(sample_nodes[i],curr_map))



            mc, me, avg, bad = distortion_row(true_dist_row.T, hyp_dist_row[:n].T.astype('float64') ,n-1,sample_nodes[i]-1)
            _wcs[i]  = mc*me

            _d_avgs[i] = avg


    maps  = sum(_maps)
    d_avg = sum(_d_avgs)
    wc    = max(_wcs)

    if weighted:
        print("Note: MAP is not well defined for weighted graphs")

    # Final stats:
    print("Final MAP = {}".format(maps/samples))
    print("Final d_avg = {}, d_wc = {}".format(d_avg/samples,wc))



#TODO main method?
#if __name__ == "__main__":
    # execute only if run as a script
#    main()
