import argparse
#import networkx as nx
import pdb
from load_graph import *
from load_graph import is_weighted
import networkx as nx
from mpmath import mp
import time
#import graph_util as gu

parser = argparse.ArgumentParser(description='parameters for the embeddings')

parser.add_argument('-d', '--dataset', metavar='dataset', type=str,
                    help='dataset to embed')

parser.add_argument('-l','--dim', metavar = 'dimension', type=int,
                    default = 3, help= 'dimension of the embeding (>2)')


parser.add_argument('-p','--precision', metavar = 'precision', type=int,
                    default = 256, help= 'precision required')


parser.add_argument('-e', '--eps', metavar = 'epsilon distortion', type=float,
                    default = 0.1, help = 'epsilon for WC_distortion distortion')


parser.add_argument('-a','--auto_tau', action='store_true',
                    help= 'Calculate scale assuming 64-bit final embedding')

parser.add_argument('-c','--code_thoery', action='store_true',
                    help= 'Use coding-theoretic child placement')


parser.add_argument('--s',  action='store_true',
                    help= 'Get statistics')


parser.add_argument('--y', action='store_true',
                    help= 'Save the distance matrix')

parser.add_argument('-m', '--save_embedding', metavar='name of saved file', type=str,
                    help= 'Save embedding to file')




parser.add_argument('--w', metavar = 'visualize',
                    help= 'Visualize the embedding (only for 2 dimensions)')

#pdb.set_trace()





#starts the embeddings
#def embeding(parser):
args = parser.parse_args()
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


######-------------LOAD THE GRAPH--------------######
# THE GRAPH MUST BE ALWAYS A TREE
#TODO: assert that the graph is always a tree
G = load_graph(args.dataset)
######-------------LOAD THE GRAPH--------------######

weighted = is_weighted(G)


print("\nGraph information")

# Number of vertices:
n = G.order();
print("Number of vertices = {}".format(n));

# Number of edges
num_edges = G.number_of_edges();
print("Number of edges = {}".format(num_edges));

root, d_max   = max_degree(G)
root = 0



# A few statistics
n_bfs   = G.order()
degrees = G.degree()

path_length  = nx.dag_longest_path_length(G)
print("Max degree = {}, Max path = {}".format(d_max,path_length))



start = time.time()

if args.auto_tau:
    mp.dps = 100
    r = 1 - mp.eps()/2
    m = mp.log((1+r)/(1-r))
    tau = m/(1.3*path_length)
elif args.eps != nothing:
    print("Epsilon  = {}".format(parsed_args["eps"]))
    epsilon = parsed_args["eps"]
    tau = get_emb_par(G, 1, epsilon, weighted)
else:
    #TODO do not allow this tau make an assertion tat ensures the user pick
    #oneof the options.
    tau = 1.0
print(tau)
# Print out the scaling factor we got
print("Scaling factor tau = ", float(tau))

use_codes = False
if args.code_thoery:
    print("Using coding theoretic child placement")
    use_codes = True
else:
    print("Using uniform sphere child placement")





use_codes = False
if args.code_thoery:
    print("Using coding theoretic child placement")
    use_codes = True
else:
    print("Using uniform sphere child placement")



if args.dim != None and args.dim != 2:
    dim = args.dim
    #T = hyp_embedding_dim(G_BFS, root, weighted, dim, tau, d_max, use_codes)
else:
    pass
    #T = hyp_embedding(G_BFS, root, weighted, tau, visualize)
end = time.time()



print( end - start)


#if __name__ == "__main__":
    # execute only if run as a script
#    main()