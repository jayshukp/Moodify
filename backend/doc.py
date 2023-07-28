import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import csv
import pandas as pd

#WORKS ALONG A PAST ADATSET OF SONGS

def isolate(df, genre_name):
    specific_value = genre_name
    df_filtered = df[df['genre'] == specific_value]
    return df_filtered

def getFinalStats(new_df, artist_name, song_name, genre_name):
    artistRow = new_df.index[new_df['artist_name'] == artist_name].tolist()
    songRow = new_df.index[new_df['track_name'] == song_name].tolist()

    if songRow:
        for i in range(len(songRow)):
            numRow = int(songRow[i])
            value = new_df.at[numRow, 'artist_name']
            if value == artist_name:
                return songRow[i]
    else:
        print("Song not found in the specified genre.")

def get_trackid(df, artist_row):
    if artist_row is not None:
        value = df.loc[artist_row, 'track_id']
        return value
    else:
        return None