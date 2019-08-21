import pandas as pd
import numpy as np
import time
import sys
import os
import gc
import networkx as nx

def edge_list_build(input_path, output_path, write_to_disk=True):
    """
    Function that reads the database and returns the edge list to be fed into
    the embedding code.

    It creates a dictionary with unique values of parent and parent-children
    relationships as a hash map and maps this relation into the input dataframe.

    Finally, saves in output file relationship between father (leftmost column)
    and children (rightmost column).

    Input:
    * input_path: String. Path of csv file where each register corresponds to a
                  parent-children relationship in the graph.
    * output_path: String. Path to save the resulting file at.
    * write_to_disk: Boolean. Whether to write file to disk or return to memory.

    Output:
    * Optionally returns to memory a tuple tree_edge_list and hash_df, corresponding
    to the tree's edge list and the hash map, respectively.
    """

    start_time = time.time()

    df = pd.read_csv(input_path, sep='\t', header=None)

    for col in range(1, len(df.columns)):
        df.iloc[:, col] = df.iloc[:, col-1] + '_' + df.iloc[:, col]

    n_divs = len(df.columns) - 1


    dict_node_names = {}

    for id, node_name in enumerate(np.unique(df.values.flatten())):
        dict_node_names[node_name] = id + 1

    tmp_df = pd.DataFrame.from_dict(dict_node_names, orient='index')
    tmp_df.reset_index(inplace=True)
    tmp_df.rename({'index': 'nodes', 0: 'hash'}, inplace=True, axis=1)

    hash_df = tmp_df['nodes'].str.split('_', n=n_divs, expand=True)
    hash_df = pd.concat([hash_df, tmp_df['hash']], axis=1)

    for col_name in df.columns:
        df[col_name] = df[col_name].map(dict_node_names)

    df['root'] = 0
    colnames = df.columns.values
    colnames = list(colnames[-1:]) + list(colnames[:-1])
    df = df[colnames]

    df_tuples = pd.DataFrame()

    for i in range(len(df.columns) - 1):
        df_tuples[i] = list(df[df.columns[i:i + 2]].itertuples(index=False, name=None))
    del df
    gc.collect()

    nodes_list = []

    for col_id in range(0, df_tuples.shape[1]):
        father_child = df_tuples.iloc[:, col_id].drop_duplicates().values
        nodes_list.extend(father_child)

    graph = nx.DiGraph(nodes_list)
    graph_bfs = nx.bfs_tree(graph, 0)

    tree_edge_list = list(graph_bfs.edges)

    if write_to_disk:
        f_out = open(os.path.join(output_path, 'edges_list.edges'), 'w')
        for t in tree_edge_list:
            line = ' '.join(str(x) for x in t)
            f_out.write(line + '\n')
        f_out.close()

        hash_df.to_csv(os.path.join(output_path, 'hash_map.csv'), index=False, sep='\t')
        end_time = time.time()
        return
    else:
        end_time = time.time()
        return tree_edge_list, hash_df


def load_graph(file_name, directed=True):
    """
    Creates a graph from a file with the list of edges.

    Input:
        * file_name: String. Path of the file with the list of edges.
        * directed: Boolean. Whether or not the graph is directed.

    Output:
        * NetworkX Graph object.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            if len(tokens) > 2:
                w = float(tokens[2])
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u,v)
    return G
