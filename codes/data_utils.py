import pandas as pd
import numpy as np 
import sqlite3
import time
import sys


def edge_list_build(db_path):
    """
    Function that reads the database and returns the edge list
    to be fed into the embedding code.

    Input: path of database file in sqlite

    Output: Written file on ../data/processed directory containing edges list of tree
    """
    start = time.time()
    """
    artist_album_song_df = pd.DataFrame({'artist': ['Coldplay', 'Coldplay', 'Coldplay', 'Foo Fighters', 'Foo Fighters', 'Foo Fighters'],
                                         'album':  ['A Rush of Blood to the Head', 'A Rush of Blood to the Head', 'Parachutes', 'Wasting Light', 'Wasting Light', 'Hits'],
                                         'song':   ['Politicks', 'Clocks', "Everything's not lost", 'Walk', 'Rope', 'Walk'] })
    """

    artist_album_song_df = pd.read_csv(db_path, sep='\t', index_col='Unnamed: 0')
    artist_album_song_df.dropna(inplace=True)
    
    print(artist_album_song_df.head(10))    
    print(50*"*")

    #sys.exit(0)


    # Parsing artists, albums and songs into ids in a dictionary. ids will begin at 1
    # so that id 0 will correspond to the root node later
    dict_node_names = {}

    #print(np.unique(artist_album_song_df.values.flatten()))

    #print(artist_album_song_df.values.flatten() )
    for id, node_name in enumerate(np.unique(artist_album_song_df.values.flatten())):
        #print(id+1, node_name)
        dict_node_names[node_name] = id+1

    #print(dict_node_names)
    print("Dictionary was created")
    print(50*"*")


    # Mapping the dictionary of ids for each column of the dataset
    for col_name in artist_album_song_df.columns:
        artist_album_song_df[col_name] = artist_album_song_df[col_name].map(dict_node_names)

    print( artist_album_song_df.head(10) )
    print("Mapping finished")
    print(50*"*")

    # -----  Creating edges list -----
    # Unique array of artists. Column 0 should correspond to those nodes
    # connected to the root node

    # unique_artist = artist_album_song_df['artist'].unique()
    unique_artist = artist_album_song_df.iloc[:, 0].unique()
    unique_artist_num = np.zeros_like(unique_artist)
    
    # Creating a dataframe with two columns, first one will be 
    # all zeros and the second one correspond to the unique values of the 
    # first nodes connected to the root
    edge_list_df = pd.DataFrame({'node_father': unique_artist_num,
                                 'node_child': unique_artist})


    #print(edge_list_df.head())
    #print( artist_album_song_df.iloc[:,[0,1]].drop_duplicates().values )
    

    # Open file connection to write:
    edge_list_out = open( './../data/processed/music.edges', 'w') # output file

    # Saving relationship between root and artists
    for id in range(len(edge_list_df)):
        edge_list_out.write(str(edge_list_df.iloc[id, 0]) + "\t" + str(edge_list_df.iloc[id, 1]) + "\n")

    
    # Saving relationship father and child nodes
    for col_id in range(0, len(artist_album_song_df.columns)-1 ):

        # for each column col_id (father) and the next column col_id+1 (child)
        # we obtain an array with the unique pair values
        #temp_df = 
        #father_child = artist_album_song_df.iloc[:,[col_id, col_id+1]].drop_duplicates(subset='song', keep="first").values        
        father_child = artist_album_song_df.iloc[:,[col_id, col_id+1]].drop_duplicates().values        
        #print(father_child)
        #print( len(father_child) )

        # Saving per each row of the numpy array with unique father-child pairs
        # father corresponds to index 0, child to index 1
        for row_id in range( len(father_child) ):
            edge_list_out.write( str(father_child[row_id, 0]) + "\t" +  str(father_child[row_id, 1])  + "\n")

    # Closing connection
    edge_list_out.close()

    end = time.time()
    print("Total time spent", end - start)

    return

if __name__ == '__main__':
    #edge_list_build(db_path='./../data/raw/musicbrainz.db')
    edge_list_build(db_path='./../data/raw/musicbrainz.csv')
    #edge_list_build(db_path='./../data/raw/musicbrainz_100.csv')
