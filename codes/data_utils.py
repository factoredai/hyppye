import pandas as pd
import numpy as np 
import sqlite3
import time
import sys


def edge_list_build(db_path):
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

    artist_album_song_df = pd.read_csv(db_path, sep='\t', index_col='Unnamed: 0')
    artist_album_song_df.dropna(inplace=True)
    
    print(artist_album_song_df.head(10))    
    print(50*"*")

    # Parsing artists, albums and songs into ids in a dictionary.
    dict_node_names = {}
   
    for id, node_name in enumerate(np.unique(artist_album_song_df.values.flatten())):        
        dict_node_names[node_name] = id+1

    print("Dictionary was created")
    print(50*"*")

    # Mapping the dictionary of ids for each column of the dataset
    for col_name in artist_album_song_df.columns:
        artist_album_song_df[col_name] = artist_album_song_df[col_name].map(dict_node_names)

    print( artist_album_song_df.head(10) )
    print("Mapping finished")
    print(50*"*")

    # Creating edges list     
    unique_artist = artist_album_song_df.iloc[:, 0].unique()
    unique_artist_num = np.zeros_like(unique_artist)
        
    edge_list_df = pd.DataFrame({'node_father': unique_artist_num,
                                 'node_child': unique_artist})

    # Open file connection to write:
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

    return

if __name__ == '__main__':
    edge_list_build(db_path='./../data/raw/musicbrainz.csv')

