import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Set up Spotify API credentials
client_id = '392db6aaf107400b943d24790c76b85f'
client_secret = '0f5ba0a9af4d428a80b3a45401920b56'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_track_id(artist_name, song_name):
    # Search for the track using the artist and song name
    query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=query, type='track', limit=1)

    # Retrieve the track ID from the search results
    if results['tracks']['items']:
        track_id = results['tracks']['items'][0]['id']
        return track_id
    else:
        print("Track not found.")
        return None

# Create a DataFrame to store the search results
columns = ['Artist', 'Song', 'Track ID']
df = pd.DataFrame(columns=columns)
