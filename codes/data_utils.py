import pandas as pd
import numpy as np
import time
import sys
import gc
import networkx as nx

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def edge_list_build(db_path, out_path, write2disk=True):
    """
    Function that reads the database and returns the edge list to be fed into 
    the embedding code. It creates a dictionary with unique values of 
    artists, album and song and maps it into the input dataframe.
    Finally, saves in output file relationship between father (leftmost column) 
    and children (rightmost column)
    
    Input: path of csv file with structure: artist, album, song

    Output: Written file on ../data/processed directory containing edges list 
            of tree
    """

    start = time.time()

    df = pd.read_csv(db_path, sep='\t', header=None)
        
    print(df.head(10))
    print(50*"*")
    print(df.tail(10))        
    print(50*"*")
        
    for col in range(1, len(df.columns)):
        df.iloc[:, col] = df.iloc[:, col-1] + '_' + df.iloc[:, col]

    n_divs = len(df.columns)-1
    
    print(df.head(10))
    print(50*"*")
    print(df.tail(10))        
    print(50*"*")
    
    dict_node_names = {}
   
    for id, node_name in enumerate(np.unique(df.values.flatten())):        
        dict_node_names[node_name] = id+1

    print("Dictionary size", len(dict_node_names))
    print(100*'*')    
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

    for i in range(len(df.columns)-1):
        df_tuples[i] = list(df[df.columns[i:i+2]].itertuples(index=False,
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

    print(len(nodes_list))
    print(nodes_list[:5])

    graph = nx.DiGraph(nodes_list)
    graph_bfs = nx.bfs_tree(graph, 0)

    tree_edge_list = list(graph_bfs.edges)
    
    if write2disk:
        f_out = open(out_path + 'music_info.edges', 'w')
        for t in tree_edge_list:
            line = ' '.join(str(x) for x in t)
            f_out.write(line + '\n')
        f_out.close()
        
        hash_df.to_csv(out_path + 'hash_map.csv', index=False, sep='\t')

        end = time.time()
        print("Total time spent in seconds:", end - start)

        return
    else:
        end = time.time()
        print("Total time spent in seconds:", end - start)
        return tree_edge_list, hash_df



    """
    for col_name in df.columns:
        df[col_name] = df[col_name].map(dict_node_names)

    print(df.head(10))
    print(df.tail(10))
    print("Mapping finished")
    print(50*"*")
   
    unique_artist = artist_album_song_df.iloc[:, 0].unique()
    unique_artist_num = np.zeros_like(unique_artist)
        
    edge_list_df = pd.DataFrame({'node_father': unique_artist_num,
                                 'node_child': unique_artist})

  
    edge_list_out = open( './../data/processed/music.edges', 'w')

    for id in range(len(edge_list_df)):
        edge_list_out.write(str(edge_list_df.iloc[id, 0]) + "\t" + str(edge_list_df.iloc[id, 1]) + "\n")

    for col_id in range(0, len(artist_album_song_df.columns)-1 ):

        father_child = artist_album_song_df.iloc[:,[col_id, col_id+1]].drop_duplicates().values        

        for row_id in range( len(father_child) ):
            edge_list_out.write( str(father_child[row_id, 0]) + "\t" +  str(father_child[row_id, 1])  + "\n")

    edge_list_out.close()

    end = time.time()
    print("Total time spent", end - start)
    """



if __name__ == '__main__':
    edge_list_build(db_path='./../data/raw/music_info.txt',
                    out_path='./../data/processed/')

