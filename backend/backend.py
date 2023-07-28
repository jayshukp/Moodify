from flask import Flask, request, jsonify
from api_retrieve import get_audio_features, detect_mood
from get_track_info import get_track_id
from lyrics import get_track_lyrics
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/detectMood', methods=['POST'])
def detect_mood_api():
    data = request.get_json()
    artist_name = data.get('artist')
    song_name = data.get('song')

    track_id = get_track_id(artist_name, song_name)
    audio_features = get_audio_features(track_id)
    lyricsString = get_track_lyrics(song_name, artist_name)

    if audio_features:
        mood, explanation, statistics = detect_mood(audio_features, lyricsString)
        response = {
            'mood': mood,
            'explanation': explanation,
            'statistics': statistics
        }
    else:
        response = {
            'error': 'Failed to retrieve audio features.'
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run('0.0.0.0', '5001')# Change the port number to an available port, e.g., 5001

