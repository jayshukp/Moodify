from api_retrieve import get_audio_features,detect_mood
from get_track_info import get_track_id
from lyrics import get_track_lyrics
import requests
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import csv
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import textblob # Sentiment analysis library







# Example usage
def main():
 
    # Example usage
    artist_name = input("Enter the artist name: ")
    song_name = input("Enter the song name: ")

    
    track_id = get_track_id(artist_name, song_name)
    audio_features = get_audio_features(track_id)
    lyricsString = get_track_lyrics(song_name, artist_name)
    
    if audio_features:
        mood, explanation, statistics = detect_mood(audio_features, lyricsString)
        print(f"Mood: {mood}")
        print(f"Explanation: {explanation}")
        print(f"Statistics:\n{statistics}")
        
    else:
        print("Failed to retrieve audio features.")

if __name__ == '__main__':
    main()
