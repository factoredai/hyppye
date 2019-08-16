import pandas as pd
import numpy as np
import time
import sys
import gc
import networkx as nx


def edge_list_build(input_path, output_path, write_to_disk=True):
    """
    Function that reads the database and returns the edge list to be fed into 
    the embedding code. It creates a dictionary with unique values of parent 
    and parent-children relationships as a hash maps and maps this relation 
    into the input dataframe.
    Finally, saves in output file relationship between father (leftmost column) 
    and children (rightmost column)
    
    Input: path of csv file where each register corresponds to a 
    parent-children relationship in the graph
    Output: Returns to memory of write two files: first return/file 
    corresponds to edge list in the tree; second return/file corresponds to  
    hash map.
    """

    start_time = time.time()

    df = pd.read_csv(input_path, sep='\t', header=None)
                
    for col in range(1, len(df.columns)):
        df.iloc[:, col] = df.iloc[:, col-1] + '_' + df.iloc[:, col]

    n_divs = len(df.columns) - 1
    
    
    dict_node_names = {}
   
    for id, node_name in enumerate(np.unique(df.values.flatten())):        
        dict_node_names[node_name] = id + 1

    print("Dictionary size", len(dict_node_names))    
    print("Dictionary was created")
    print(50*"*")
    
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
    
    print("Mapping finished")
    print(50*"*")

    df_tuples = pd.DataFrame()

    for i in range(len(df.columns) - 1):
        df_tuples[i] = list(df[df.columns[i:i + 2]].itertuples(index=False,
                                                               name=None))
    
    del df
    gc.collect()

    print(df_tuples.head(10))
    print(df_tuples.tail(10))    
    print(50*"*")

    nodes_list = []

    for col_id in range(0, df_tuples.shape[1]):
        father_child = df_tuples.iloc[:, col_id].drop_duplicates().values
        nodes_list.extend(father_child)

    print("Size of nodes list", len(nodes_list))    

    graph = nx.DiGraph(nodes_list)
    graph_bfs = nx.bfs_tree(graph, 0)

    tree_edge_list = list(graph_bfs.edges)
    
    if write_to_disk:
        f_out = open(output_path + 'music_info.edges', 'w')
        for t in tree_edge_list:
            line = ' '.join(str(x) for x in t)
            f_out.write(line + '\n')
        f_out.close()
        
        hash_df.to_csv(output_path + 'hash_map.csv', index=False, sep='\t')

        end_time = time.time()
        print("Total time spent in seconds:", end_time - start_time)

        return
    else:
        end_time = time.time()
        print("Total time spent in seconds:", end_time - start_time)
        return tree_edge_list, hash_df
