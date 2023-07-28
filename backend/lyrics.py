import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius

# Set up Spotify API credentials
client_id = '392db6aaf107400b943d24790c76b85f'
client_secret = '0f5ba0a9af4d428a80b3a45401920b56'
genius_access_token = 'rW-jT6gS4_n3AmkAOcKKlJtr4tVs40lgbc2VUPiog868xO8njtqU7bk6dIGO_HzG'

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to retrieve the lyrics of a specific Spotify track
def get_track_lyrics(track_name, artist_name):
    # Search for the track using the Spotify API
    results = sp.search(q=f'track:{track_name} artist:{artist_name}', type='track', limit=1)
    
    if results['tracks']['items']:
        # Get the Spotify track ID
        track_id = results['tracks']['items'][0]['id']

        # Create a LyricsGenius client instance
        genius = lyricsgenius.Genius(genius_access_token)

        # Search for lyrics based on the track and artist name
        song = genius.search_song(track_name, artist_name)

        if song:
            return song.lyrics
        else:
            print("Lyrics not found for the given track.")
            return None
    else:
        print("Track not found.")
        return None

