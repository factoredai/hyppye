using PyCall

#grant visibility to local directory scripts
py"""import sys
sys.path.insert(0, "/home/jgutierrez/Lacuna/projects/hyperbolicEmbeddings_team4/hyp_embedding_dim")
"""
#import local directory script functions
hyp_to_euc_dist = pyimport("graph_utils").hyp_to_euc_dist
place_children_codes = pyimport("graph_utils").place_children_codes

#import python packages
@pyimport networkx as nx

#define variables
use_codes = true
tau = 1.0
root = 0
weighted = false #if the graph is weighted or not
dim = 100 #Hr
d_max = 30 #maximum degree of graph
G = nx.DiGraph([(0,1),(0,2),(0,3),
                (1,4),(2,5),(3,6),(3,7),
                (4,8),(5,9),(6,10),(7,11),(7,12)])#NetworkX Digraph
G_BFS = nx.bfs_tree(G, 0)

n = G_BFS.order()
T = zeros(BigFloat, n, dim)
root_children = collect(G_BFS.successors(root))
d = length(root_children)
edge_lengths = hyp_to_euc_dist(tau*ones(d, 1))

v = Int(ceil(log2(d_max)))

if use_codes
    # new way: generate a bunch of generator matrices we'll use for our codes
    Gen_matrices = Array{Array{Float64, 2}}(undef, v)
    for i=2:v
        n = 2^i-1
        H = zeros(i,n)
        for j=1:2^i-1
            h_col = digits(j, base=2, pad=i)
            H[:,j] = h_col
        end
        Gen_matrices[i] = H
    end
end

if !use_codes || d_max > dim
    print('Forbidden!')

function place_children_codes_(dim, c, Gen_matrices)
    r = Int(ceil(log2(c)))
    n = 2^r-1

    G = Gen_matrices[r]

    # generate the codewords using our matrix
    C = zeros(BigFloat, c, dim)
    for i=0:c-1
        # codeword generated from matrix G:
        cw = (digits(i,base=2,pad=r)'*G).%(2)

        rep = Int(floor(dim/n))
        for j=1:rep
            # repeat it as many times as we can
            C[i+1,(j-1)*n+1:j*n] = big.(cw');
        end
        rm = dim-rep*n
        if rm > 0
            C[i+1,rep*n+1:dim] = big.(cw[1:rm]')
        end
    end

    # inscribe the unit hypercube vertices into unit hypersphere
    points = (big(1)/sqrt(dim)*(-1).^C)'

    return points
end

if use_codes && d <= dim
    R = place_children_codes_(dim, d, Gen_matrices)
end

R

"""


    # place the children of the root:
    if use_codes && d <= dim
        R = place_children_codes(dim, d, false, 0, Gen_matrices)
    else
        R = place_children(dim, d, false, 0, true, SB)
    end

    R = R'
    for i=1:d
        R[i,:] *= edge_lengths[i]
    end

    for i=1:d
         T[1+root_children[i],:] = R[i,:]
    end

    # queue containing the nodes whose children we're placing
    q = [];
    append!(q, root_children)
    node_idx = 0

    while length(q) > 0
        h            = q[1];
        node_idx     += 1
        if node_idx%100 == 0
            println("Placing children of node $(node_idx)")
        end

        children     = collect(G_BFS[:successors](h));
        parent       = collect(G_BFS[:predecessors](h));
        num_children = length(children);
        edge_lengths = hyp_to_euc_dist(tau*ones(num_children,1));

        append!(q, children)

        if weighted
            k = 1;
            for child in children
                weight = G_BFS[h+1][child+1]["weight"];
                edge_lengths[k] = hyp_to_euc_dist(big(weight)*tau);
                k = k+1;
            end
        end

        if num_children > 0
            if use_codes && num_children+1 <= dim
                R = add_children_dim(T[parent[1]+1,:], T[h+1,:], dim, edge_lengths, true, 0, Gen_matrices)
            else
                R = add_children_dim(T[parent[1]+1,:], T[h+1,:], dim, edge_lengths, false, SB, 0)
            end

            for i=1:num_children
                T[children[i]+1,:] = R[i,:];
            end
        end

        deleteat!(q, 1)
    end

    return T
end
"""
