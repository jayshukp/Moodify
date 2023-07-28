import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import textblob # Sentiment analysis library
import torch
from transformers import BertTokenizer, BertForSequenceClassification


# Set up Spotify API credentials
client_id = '392db6aaf107400b943d24790c76b85f'
client_secret = '0f5ba0a9af4d428a80b3a45401920b56'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to retrieve audio features for a given track ID
def get_audio_features(track_id):
    features = sp.audio_features([track_id])
    return features[0] if features else None



# Load BERT sentiment analysis model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)
model.eval()

def detect_mood(audio_features, lyrics):
    # Extract audio features
    valence = audio_features['valence']
    energy = audio_features['energy']
    danceability = audio_features['danceability']
    acousticness = audio_features['acousticness']
    instrumentalness = audio_features['instrumentalness']
    loudness = audio_features['loudness']
    speechiness = audio_features['speechiness']
    tempo = audio_features['tempo']

    # Perform sentiment analysis on lyrics using BERT
    inputs = tokenizer.encode_plus(lyrics, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)[0]
        sentiment = torch.softmax(logits, dim=1)[0][1].item()  # Positive sentiment probability


    # Define mood detection criteria based on the audio features and sentiment analysis
    mood = ""
    explanation = ""
    statistics = ""

    if valence > 0.8 and energy > 0.8 and sentiment > 0.6:
        mood = 'Energetic and Blissful'
        explanation = "The song has very high valence and energy, creating an energetic and blissful mood."
    # Add more specific moods below
    elif valence < 0.1 and energy > 0.8 and sentiment < -0.6:
        mood = 'Dark and Agonizing'
        explanation = "The song has very low valence and high energy, evoking feelings of darkness and agony."
    elif valence > 0.8 and energy < 0.2 and sentiment > 0.6:
        mood = 'Serene and Enchanting'
        explanation = "The song has very high valence and low energy, creating a serene and enchanting atmosphere."
    elif valence > 0.6 and energy > 0.8 and danceability > 0.8:
        mood = 'Exhilarating and Groovy'
        explanation = "The song has high valence, energy, and danceability, resulting in an exhilarating and groovy mood."
    elif valence > 0.8 and energy > 0.8 and sentiment < -0.6:
        mood = 'Energetic and Ferocious'
        explanation = "The song has very high valence and energy, but the lyrics convey ferocity or aggression."
    elif valence < 0.2 and energy > 0.8 and sentiment > 0.4:
        mood = 'Sad but Uplifting'
        explanation = "The song has very low valence and high energy, but the lyrics carry uplifting or positive sentiments."
    elif valence > 0.8 and energy < 0.2 and sentiment < -0.6:
        mood = 'Tranquil and Melancholic'
        explanation = "The song has very high valence and low energy, creating a tranquil and melancholic mood."
    elif acousticness > 0.8 and instrumentalness > 0.8:
        mood = 'Enveloping and Instrumental'
        explanation = "The song has high acousticness and instrumentalness, resulting in an enveloping and instrumental mood."
    elif loudness < -8 and speechiness > 0.6:
        mood = 'Boisterous and Talkative'
        explanation = "The song has low loudness and high speechiness, creating a boisterous and talkative mood."
    elif tempo < 80 and valence < 0.4:
        mood = 'Languid and Somber'
        explanation = "The song has a slow tempo and low valence, evoking a languid and somber atmosphere."
    elif valence > 0.8 and energy > 0.6:
        mood = 'Energetic and Exciting'
        explanation = "The song has high valence and energy, creating an energetic and exciting mood."
    elif valence < 0.2 and energy < 0.4:
        mood = 'Sad and Mellow'
        explanation = "The song has low valence and energy, evoking feelings of sadness and mellowness."
    elif danceability > 0.8:
        mood = 'Upbeat and Danceable'
        explanation = "The song has high danceability, contributing to an upbeat and danceable mood."
    # Additional Moods
    elif valence > 0.9 and energy > 0.7 and sentiment > 0.7:
        mood = 'Blissful and Ecstatic'
        explanation = "The song has extremely high valence, energy, and positive sentiment, creating a blissful and ecstatic mood."
    elif valence < 0.1 and energy > 0.7 and sentiment < -0.7:
        mood = 'Gloomy and Tormented'
        explanation = "The song has extremely low valence and high energy, evoking feelings of gloominess and torment."
    elif valence > 0.9 and energy < 0.1 and sentiment > 0.7:
        mood = 'Tranquil and Enthralling'
        explanation = "The song has extremely high valence and low energy, creating a tranquil and enthralling atmosphere."
    elif valence > 0.7 and energy > 0.9 and danceability > 0.9:
        mood = 'Euphoric and Infectious'
        explanation = "The song has very high valence, energy, and danceability, resulting in a euphoric and infectious mood."
    elif valence > 0.9 and energy > 0.9 and sentiment < -0.7:
        mood = 'Energetic and Savage'
        explanation = "The song has very high valence and energy, but the lyrics convey savagery or aggression."
    elif valence < 0.1 and energy > 0.9 and sentiment > 0.5:
        mood = 'Devastating yet Empowering'
        explanation = "The song has extremely low valence and very high energy, but the lyrics carry empowering or uplifting sentiments."
    elif valence > 0.9 and energy < 0.1 and sentiment < -0.7:
        mood = 'Serenading and Melancholic'
        explanation = "The song has extremely high valence and very low energy, creating a serenading and melancholic mood."
    elif acousticness > 0.9 and instrumentalness > 0.9:
        mood = 'Atmospheric and Serene'
        explanation = "The song has very high acousticness and instrumentalness, resulting in an atmospheric and serene mood."
    elif loudness < -10 and speechiness > 0.7:
        mood = 'Vociferous and Expressive'
        explanation = "The song has very low loudness and high speechiness, creating a vociferous and expressive mood."
    elif tempo < 60 and valence < 0.3:
        mood = 'Lethargic and Dreary'
        explanation = "The song has an extremely slow tempo and low valence, evoking a lethargic and dreary atmosphere."
    # Add more specific moods below
    elif valence > 0.9 and energy > 0.9 and sentiment > 0.9:
        mood = 'Ecstatic and Transcendent'
        explanation = "The song has extremely high valence, energy, and positive sentiment, creating an ecstatic and transcendent mood."
    elif valence < 0.1 and energy > 0.9 and sentiment < -0.9:
        mood = 'Desolate and Tortured'
        explanation = "The song has extremely low valence and high energy, evoking feelings of desolation and torment."
    elif valence > 0.9 and energy < 0.1 and sentiment > 0.9:
        mood = 'Enthralling and Mystical'
        explanation = "The song has extremely high valence and low energy, creating an enthralling and mystical atmosphere."
    elif valence > 0.8 and energy > 0.9 and danceability > 0.9:
        mood = 'Euphoric and Irresistible'
        explanation = "The song has very high valence, energy, and danceability, resulting in a euphoric and irresistible mood."
    elif valence > 0.9 and energy > 0.9 and sentiment < -0.9:
        mood = 'Energetic and Fierce'
        explanation = "The song has very high valence and energy, but the lyrics convey fierceness or aggression."
    elif valence < 0.1 and energy > 0.9 and sentiment > 0.7:
        mood = 'Heartbreaking yet Empowering'
        explanation = "The song has extremely low valence and very high energy, but the lyrics carry empowering or uplifting sentiments."
    elif valence > 0.9 and energy < 0.1 and sentiment < -0.9:
        mood = 'Enchanting and Melancholic'
        explanation = "The song has extremely high valence and very low energy, creating an enchanting and melancholic mood."
    elif acousticness > 0.9 and instrumentalness > 0.9:
        mood = 'Enveloping and Ethereal'
        explanation = "The song has very high acousticness and instrumentalness, resulting in an enveloping and ethereal mood."
    elif loudness < -12 and speechiness > 0.8:
        mood = 'Powerful and Commanding'
        explanation = "The song has very low loudness and high speechiness, creating a powerful and commanding mood."
    elif tempo < 50 and valence < 0.2:
        mood = 'Hypnotic and Haunting'
        explanation = "The song has an extremely slow tempo and low valence, evoking a hypnotic and haunting atmosphere."
    # Add more specific moods below
    # Additional Moods
    elif valence > 0.7 and energy > 0.7 and sentiment > 0.5:
        mood = 'Joyful and Radiant'
        explanation = "The song has high valence, energy, and positive sentiment, creating a joyful and radiant mood."
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.5:
        mood = 'Melancholic and Reflective'
        explanation = "The song has low valence and high energy, evoking a melancholic and reflective mood."
    elif valence > 0.7 and energy < 0.3 and sentiment > 0.5:
        mood = 'Dreamy and Enigmatic'
        explanation = "The song has high valence and low energy, creating a dreamy and enigmatic atmosphere."
    elif valence > 0.5 and energy > 0.7 and danceability > 0.7:
        mood = 'Lively and Rhythmic'
        explanation = "The song has high valence, energy, and danceability, resulting in a lively and rhythmic mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.5:
        mood = 'Intense and Aggressive'
        explanation = "The song has high valence and energy, but the lyrics convey intensity and aggression."
    elif valence < 0.3 and energy > 0.7 and sentiment > 0.3:
        mood = 'Sad and Hopeful'
        explanation = "The song has low valence and high energy, but the lyrics carry a sense of hope and optimism."
    # Add more specific moods below
    # Additional Moods
    elif valence > 0.7 and energy > 0.7 and sentiment > 0.5:
        mood = 'Joyful and Radiant'
        explanation = "The song has high valence, energy, and positive sentiment, creating a joyful and radiant mood."
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.5:
        mood = 'Melancholic and Reflective'
        explanation = "The song has low valence and high energy, evoking a melancholic and reflective mood."
    elif valence > 0.7 and energy < 0.3 and sentiment > 0.5:
        mood = 'Dreamy and Enigmatic'
        explanation = "The song has high valence and low energy, creating a dreamy and enigmatic atmosphere."
    elif valence > 0.5 and energy > 0.7 and danceability > 0.7:
        mood = 'Lively and Rhythmic'
        explanation = "The song has high valence, energy, and danceability, resulting in a lively and rhythmic mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.5:
        mood = 'Intense and Aggressive'
        explanation = "The song has high valence and energy, but the lyrics convey intensity and aggression."
    elif valence < 0.3 and energy > 0.7 and sentiment > 0.3:
        mood = 'Sad and Hopeful'
        explanation = "The song has low valence and high energy, but the lyrics carry a sense of hope and optimism."
    elif valence > 0.7 and energy < 0.3 and sentiment < -0.5:
        mood = 'Enchanting and Mystical'
        explanation = "The song has high valence and low energy, creating an enchanting and mystical mood."
    elif valence < 0.3 and energy < 0.3 and sentiment > 0.5:
        mood = 'Nostalgic and Sentimental'
        explanation = "The song has low valence and energy, evoking feelings of nostalgia and sentimentality."
    elif valence > 0.5 and energy > 0.7 and danceability < 0.3:
        mood = 'Tense and Brooding'
        explanation = "The song has high valence, energy, but low danceability, creating a tense and brooding mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.3:
        mood = 'Energetic and Rebellious'
        explanation = "The song has high valence and energy, but the lyrics convey a sense of rebellion."
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.3:
        mood = 'Desolate and Painful'
        explanation = "The song has low valence and high energy, evoking feelings of desolation and pain."
    elif valence > 0.7 and energy < 0.3 and sentiment > 0.3:
        mood = 'Whimsical and Playful'
        explanation = "The song has high valence and low energy, creating a whimsical and playful mood."
    elif valence < 0.3 and energy < 0.3 and sentiment < -0.5:
        mood = 'Grim and Menacing'
        explanation = "The song has low valence and energy, conveying a sense of grimness and menace."
    elif valence > 0.5 and energy > 0.7 and danceability > 0.7:
        mood = 'Uplifting and Energetic'
        explanation = "The song has high valence, energy, and danceability, creating an uplifting and energetic mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.3:
        mood = 'Passionate and Fiery'
        explanation = "The song has high valence and energy, with lyrics expressing passion and intensity."
    elif valence < 0.3 and energy > 0.7 and sentiment > 0.5:
        mood = 'Sorrowful yet Resilient'
        explanation = "The song has low valence and high energy, but the lyrics carry a sense of resilience amidst sorrow."
    elif valence > 0.7 and energy < 0.3 and sentiment < -0.3:
        mood = 'Mysterious and Intriguing'
        explanation = "The song has high valence and low energy, evoking a sense of mystery and intrigue."
    elif valence < 0.3 and energy < 0.3 and sentiment > 0.3:
        mood = 'Thoughtful and Contemplative'
        explanation = "The song has low valence and energy, encouraging thoughtful and contemplative mood."
    elif valence > 0.5 and energy > 0.7 and danceability < 0.5:
        mood = 'Calm and Serene'
        explanation = "The song has high valence and energy, but low danceability, creating a calm and serene mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.3:
        mood = 'Fiery and Assertive'
        explanation = "The song has high valence and energy, with lyrics conveying assertiveness and intensity."
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.3:
        mood = 'Haunting and Eerie'
        explanation = "The song has low valence and high energy, evoking a haunting and eerie atmosphere."
    elif valence > 0.7 and energy < 0.3 and sentiment > 0.3:
        mood = 'Enthusiastic and Optimistic'
        explanation = "The song has high valence and low energy, creating an enthusiastic and optimistic mood."
    elif valence < 0.3 and energy < 0.3 and sentiment < -0.5:
        mood = 'Oppressive and Dystopian'
        explanation = "The song has low valence and energy, conveying a sense of oppression and dystopia."
    elif valence > 0.5 and energy > 0.7 and danceability > 0.5:
        mood = 'Euphoric and Upbeat'
        explanation = "The song has high valence, energy, and danceability, resulting in a euphoric and upbeat mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.3:
        mood = 'Dynamic and Powerful'
        explanation = "The song has high valence and energy, with lyrics conveying dynamic and powerful emotions."
    elif valence < 0.3 and energy > 0.7 and sentiment > 0.5:
        mood = 'Heartbreaking yet Inspiring'
        explanation = "The song has low valence and high energy, but the lyrics carry inspiring and uplifting sentiments."
    elif valence > 0.7 and energy < 0.3 and sentiment < -0.3:
        mood = 'Enigmatic and Hypnotic'
        explanation = "The song has high valence and low energy, creating an enigmatic and hypnotic atmosphere."
    elif valence < 0.3 and energy < 0.3 and sentiment > 0.3:
        mood = 'Melancholic and Thought-provoking'
        explanation = "The song has low valence and energy, evoking a melancholic and thought-provoking mood."
    elif valence > 0.5 and energy > 0.7 and danceability < 0.7:
        mood = 'Relaxed and Chill'
        explanation = "The song has high valence, energy, but low danceability, creating a relaxed and chill mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.3:
        mood = 'Passionate and Evocative'
        explanation = "The song has high valence and energy, with lyrics evoking strong passion and emotion."
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.3:
        mood = 'Despairing and Tragic'
        explanation = "The song has low valence and high energy, conveying feelings of despair and tragedy."
    elif valence > 0.7 and energy < 0.3 and sentiment > 0.3:
        mood = 'Fantastical and Whimsical'
        explanation = "The song has high valence and low energy, creating a fantastical and whimsical mood."
    elif valence < 0.3 and energy < 0.3 and sentiment < -0.5:
        mood = 'Gloomy and Foreboding'
        explanation = "The song has low valence and energy, evoking a sense of gloominess and foreboding."
    # Add more specific moods below
    # Sadness Moods
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.5:
        mood = 'Heartbreaking and Sorrowful'
        explanation = "The song has low valence and high energy, evoking feelings of heartbreak and sorrow."
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.3:
        mood = 'Melancholic and Wistful'
        explanation = "The song has low valence and high energy, creating a melancholic and wistful mood."
    elif valence < 0.3 and energy > 0.7 and sentiment < -0.1:
        mood = 'Somber and Reflective'
        explanation = "The song has low valence and high energy, evoking a somber and reflective atmosphere."
    elif valence < 0.3 and energy > 0.7 and sentiment < 0.1:
        mood = 'Pensive and Contemplative'
        explanation = "The song has low valence and high energy, encouraging a pensive and contemplative mood."
    elif valence < 0.3 and energy > 0.7 and sentiment > 0.1:
        mood = 'Sad but Resilient'
        explanation = "The song has low valence and high energy, but the lyrics convey resilience amidst sadness."
    elif valence < 0.3 and energy > 0.5 and sentiment < -0.5:
        mood = 'Desolate and Loneliness'
        explanation = "The song has low valence, high energy, and lyrics that express feelings of desolation and loneliness."
    elif valence < 0.3 and energy > 0.5 and sentiment < -0.3:
        mood = 'Grief-stricken and Despair'
        explanation = "The song has low valence, high energy, and conveys a sense of grief and despair."
    elif valence < 0.3 and energy > 0.5 and sentiment < -0.1:
        mood = 'Tearful and Sullen'
        explanation = "The song has low valence, high energy, and evokes a tearful and sullen mood."
    elif valence < 0.3 and energy > 0.5 and sentiment > 0.1:
        mood = 'Sadness with a Glimmer of Hope'
        explanation = "The song has low valence, high energy, but the lyrics carry a glimmer of hope amidst sadness."
    elif valence < 0.3 and energy > 0.5 and sentiment > 0.3:
        mood = 'Melancholic yet Uplifting'
        explanation = "The song has low valence, high energy, and conveys a sense of melancholy but with uplifting elements."
    # Add more specific moods below
    # Excitement and Happiness Moods
    elif valence > 0.8 and energy > 0.8 and sentiment > 0.7:
        mood = 'Ecstatic and Joyful'
        explanation = "The song has very high valence, energy, and positive sentiment, creating an ecstatic and joyful mood."
    elif valence > 0.8 and energy > 0.8 and sentiment > 0.5:
        mood = 'Euphoric and Exhilarating'
        explanation = "The song has very high valence, energy, and positive sentiment, resulting in a euphoric and exhilarating mood."
    elif valence > 0.8 and energy > 0.8 and sentiment > 0.3:
        mood = 'Upbeat and Radiant'
        explanation = "The song has very high valence, energy, and positive sentiment, evoking an upbeat and radiant atmosphere."
    elif valence > 0.8 and energy > 0.8 and sentiment > 0.1:
        mood = 'Cheerful and Optimistic'
        explanation = "The song has very high valence, energy, and positive sentiment, creating a cheerful and optimistic mood."
    elif valence > 0.8 and energy > 0.8 and sentiment < 0.1:
        mood = 'Happy but Bittersweet'
        explanation = "The song has very high valence and energy, but the lyrics convey a touch of bittersweetness."
    elif valence > 0.8 and energy > 0.8 and sentiment < -0.1:
        mood = 'Vibrant and Nostalgic'
        explanation = "The song has very high valence and energy, evoking a vibrant and nostalgic mood."
    elif valence > 0.8 and energy > 0.6 and danceability > 0.8:
        mood = 'Uplifting and Dance-Pop'
        explanation = "The song has high valence, energy, and danceability, resulting in an uplifting and dance-pop mood."
    elif valence > 0.8 and energy > 0.6 and danceability > 0.6:
        mood = 'Feel-Good and Catchy'
        explanation = "The song has high valence, energy, and danceability, creating a feel-good and catchy atmosphere."
    elif valence > 0.8 and energy > 0.6 and danceability > 0.4:
        mood = 'Bright and Energetic'
        explanation = "The song has high valence, energy, and danceability, evoking a bright and energetic mood."
    elif valence > 0.8 and energy > 0.6 and danceability < 0.4:
        mood = 'Excited but Chill'
        explanation = "The song has high valence and energy, but the lyrics create a chill and relaxed vibe."
    # Add more specific moods below
    # Complex Emotional Moods
    elif valence > 0.6 and energy > 0.6 and sentiment > 0.5:
        mood = 'Bittersweet and Reflective'
        explanation = "The song has high valence, energy, and positive sentiment, creating a bittersweet and reflective mood."
    elif valence > 0.6 and energy > 0.6 and sentiment < -0.5:
        mood = 'Melancholic and Thoughtful'
        explanation = "The song has high valence and energy, but the lyrics convey a melancholic and thoughtful atmosphere."
    elif valence > 0.6 and energy > 0.6 and sentiment > 0.3:
        mood = 'Nostalgic and Hopeful'
        explanation = "The song has high valence, energy, and positive sentiment, evoking a nostalgic yet hopeful mood."
    elif valence > 0.6 and energy > 0.6 and sentiment < -0.3:
        mood = 'Yearning and Contemplative'
        explanation = "The song has high valence and energy, but the lyrics evoke a sense of yearning and contemplation."
    elif valence > 0.6 and energy > 0.6 and sentiment > 0.1:
        mood = 'Whimsical and Sentimental'
        explanation = "The song has high valence, energy, and positive sentiment, creating a whimsical and sentimental mood."
    elif valence > 0.6 and energy > 0.6 and sentiment < -0.1:
        mood = 'Ambiguous and Enigmatic'
        explanation = "The song has high valence and energy, but the lyrics convey an ambiguous and enigmatic mood."
    elif valence > 0.6 and energy > 0.4 and danceability > 0.6:
        mood = 'Euphoric and Introspective'
        explanation = "The song has high valence, energy, and danceability, resulting in a euphoric and introspective mood."
    elif valence > 0.6 and energy > 0.4 and danceability > 0.4:
        mood = 'Enthralling and Evocative'
        explanation = "The song has high valence, energy, and danceability, evoking an enthralling and evocative atmosphere."
    elif valence > 0.6 and energy > 0.4 and danceability < 0.4:
        mood = 'Mysterious and Intriguing'
        explanation = "The song has high valence and energy, but the lyrics create a mysterious and intriguing mood."
    # Add more specific moods below
    # Sadness Moods
    elif valence < 0.4 and energy < 0.4 and sentiment < -0.3:
        mood = 'Melancholic and Wistful'
        explanation = "The song has low valence, energy, and negative sentiment, evoking a melancholic and wistful mood."
    elif valence < 0.4 and energy < 0.4 and sentiment < -0.1:
        mood = 'Gloomy and Reflective'
        explanation = "The song has low valence, energy, and negative sentiment, creating a gloomy and reflective atmosphere."
    elif valence < 0.4 and energy < 0.4 and sentiment > 0.1:
        mood = 'Tender and Sorrowful'
        explanation = "The song has low valence and energy, but the lyrics convey tenderness and sorrow."
    elif valence < 0.4 and energy < 0.4 and sentiment > 0.3:
        mood = 'Bittersweet and Nostalgic'
        explanation = "The song has low valence and energy, but the lyrics evoke a bittersweet and nostalgic mood."
    elif valence < 0.4 and energy > 0.4 and sentiment < -0.3:
        mood = 'Sad and Dramatic'
        explanation = "The song has low valence and high energy, creating a sad and dramatic mood."
    elif valence < 0.4 and energy > 0.4 and sentiment < -0.1:
        mood = 'Heartbroken and Yearning'
        explanation = "The song has low valence and high energy, evoking feelings of heartbreak and yearning."
    elif valence < 0.4 and energy > 0.4 and sentiment > 0.1:
        mood = 'Melancholic and Contemplative'
        explanation = "The song has low valence and high energy, creating a melancholic and contemplative atmosphere."
    elif valence < 0.4 and energy > 0.4 and sentiment > 0.3:
        mood = 'Sad and Hopeful'
        explanation = "The song has low valence and high energy, but the lyrics convey a sense of hope amid sadness."
        # Excitement Moods
    elif valence > 0.7 and energy > 0.7 and sentiment > 0.5:
        mood = 'Energetic and Ecstatic'
        explanation = "The song has high valence, energy, and positive sentiment, creating an energetic and ecstatic mood."
    elif valence > 0.7 and energy > 0.7 and sentiment > 0.3:
        mood = 'Thrilling and Exhilarating'
        explanation = "The song has high valence, energy, and positive sentiment, evoking a thrilling and exhilarating mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.3:
        mood = 'Intense and Explosive'
        explanation = "The song has high valence, energy, but the lyrics convey intensity and explosiveness."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.1:
        mood = 'Passionate and Fiery'
        explanation = "The song has high valence, energy, but the lyrics evoke passion and fiery emotions."
    elif valence > 0.7 and energy > 0.7 and sentiment > 0.1:
        mood = 'Joyful and Vibrant'
        explanation = "The song has high valence, energy, and positive sentiment, creating a joyful and vibrant atmosphere."
    elif valence > 0.7 and energy > 0.7 and sentiment > 0.3:
        mood = 'Euphoric and Exuberant'
        explanation = "The song has high valence, energy, and positive sentiment, evoking a euphoric and exuberant mood."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.3:
        mood = 'Wild and Unrestrained'
        explanation = "The song has high valence, energy, but the lyrics convey wildness and unrestrained emotions."
    elif valence > 0.7 and energy > 0.7 and sentiment < -0.1:
        mood = 'Hypnotic and Thrilling'
        explanation = "The song has high valence, energy, but the lyrics evoke a hypnotic and thrilling mood."
        # Memories and Reminiscence Moods
    elif valence > 0.5 and energy < 0.4 and sentiment > 0.4:
        mood = 'Nostalgic and Sentimental'
        explanation = "The song has high valence, low energy, and positive sentiment, evoking a nostalgic and sentimental mood."
    elif valence > 0.5 and energy < 0.4 and sentiment > 0.2:
        mood = 'Reflective and Melancholic'
        explanation = "The song has high valence, low energy, and positive sentiment, creating a reflective and melancholic atmosphere."
    elif valence > 0.5 and energy < 0.4 and sentiment < -0.4:
        mood = 'Yearning and Longing'
        explanation = "The song has high valence, low energy, but the lyrics convey a sense of yearning and longing."
    elif valence > 0.5 and energy < 0.4 and sentiment < -0.2:
        mood = 'Nostalgic and Bittersweet'
        explanation = "The song has high valence, low energy, but the lyrics evoke a nostalgic and bittersweet mood."
    elif valence > 0.5 and energy > 0.4 and sentiment > 0.4:
        mood = 'Sentimental and Reflective'
        explanation = "The song has high valence and energy, evoking a sentimental and reflective mood."
    elif valence > 0.5 and energy > 0.4 and sentiment > 0.2:
        mood = 'Ethereal and Dreamy'
        explanation = "The song has high valence and energy, creating an ethereal and dreamy atmosphere."
    elif valence > 0.5 and energy > 0.4 and sentiment < -0.4:
        mood = 'Yearning and Nostalgic'
        explanation = "The song has high valence and energy, but the lyrics convey a sense of yearning and nostalgia."
    elif valence > 0.5 and energy > 0.4 and sentiment < -0.2:
        mood = 'Melancholic and Reflective'
        explanation = "The song has high valence and energy, evoking a melancholic and reflective mood."
        # Loving and Caring Moods
    elif valence > 0.7 and energy > 0.5 and sentiment > 0.5:
        mood = 'Romantic and Affectionate'
        explanation = "The song has high valence, energy, and positive sentiment, evoking a romantic and affectionate mood."
    elif valence > 0.7 and energy > 0.5 and sentiment > 0.2:
        mood = 'Warm and Tender'
        explanation = "The song has high valence, energy, and positive sentiment, creating a warm and tender atmosphere."
    elif valence > 0.7 and energy > 0.5 and sentiment < -0.5:
        mood = 'Longing and Devotion'
        explanation = "The song has high valence, energy, but the lyrics convey a sense of longing and devotion."
    elif valence > 0.7 and energy > 0.5 and sentiment < -0.2:
        mood = 'Passionate and Tender'
        explanation = "The song has high valence, energy, evoking a passionate and tender mood."
    elif valence > 0.7 and energy < 0.5 and sentiment > 0.5:
        mood = 'Gentle and Compassionate'
        explanation = "The song has high valence, low energy, and positive sentiment, creating a gentle and compassionate atmosphere."
    elif valence > 0.7 and energy < 0.5 and sentiment > 0.2:
        mood = 'Loving and Serene'
        explanation = "The song has high valence, low energy, and positive sentiment, evoking a loving and serene mood."
    elif valence > 0.7 and energy < 0.5 and sentiment < -0.5:
        mood = 'Yearning and Affectionate'
        explanation = "The song has high valence, low energy, but the lyrics convey a sense of yearning and affection."
    elif valence > 0.7 and energy < 0.5 and sentiment < -0.2:
        mood = 'Tender and Nurturing'
        explanation = "The song has high valence, low energy, evoking a tender and nurturing mood."
        # Complex Loving and Caring Moods
    elif valence > 0.7 and energy > 0.5 and sentiment > 0.5:
        mood = 'Bittersweet Love'
        explanation = "The song has high valence, energy, and positive sentiment, evoking a bittersweet feeling of love."
    elif valence > 0.7 and energy > 0.5 and sentiment < -0.5:
        mood = 'Complicated Devotion'
        explanation = "The song has high valence, energy, but the lyrics convey a sense of complicated devotion and affection."
    elif valence > 0.7 and energy < 0.5 and sentiment > 0.5:
        mood = 'Turbulent Affection'
        explanation = "The song has high valence, low energy, and positive sentiment, reflecting a turbulent yet affectionate mood."
    elif valence > 0.7 and energy < 0.5 and sentiment < -0.5:
        mood = 'Conflicting Love'
        explanation = "The song has high valence, low energy, but the lyrics portray conflicting emotions within love and care."
    elif valence > 0.6 and energy > 0.8 and sentiment > 0.4:
        mood = 'Intense Passion'
        explanation = "The song has high valence, energy, and positive sentiment, representing an intense and passionate love."
    elif valence > 0.6 and energy > 0.8 and sentiment < -0.4:
        mood = 'Fiery Devotion'
        explanation = "The song has high valence, energy, but the lyrics convey a fiery and intense sense of devotion."
    elif valence > 0.6 and energy < 0.2 and sentiment > 0.4:
        mood = 'Gentle Adoration'
        explanation = "The song has high valence, low energy, and positive sentiment, expressing a gentle and adoring love."
    elif valence > 0.6 and energy < 0.2 and sentiment < -0.4:
        mood = 'Tender Turmoil'
        explanation = "The song has high valence, low energy, but the lyrics depict a tender yet tumultuous emotional state."
        # Moods of Confusion in Life & Love
    elif valence < 0.4 and energy > 0.6 and sentiment > 0.4:
        mood = 'Lost in Uncertainty'
        explanation = "The song has low valence, high energy, and positive sentiment, reflecting a sense of being lost in uncertainty."
    elif valence < 0.4 and energy > 0.6 and sentiment < -0.4:
        mood = 'Chaotic Heart'
        explanation = "The song has low valence, high energy, but the lyrics convey a chaotic and confused state of the heart."
    elif valence < 0.4 and energy < 0.4 and sentiment > 0.4:
        mood = 'Confused Longing'
        explanation = "The song has low valence, low energy, and positive sentiment, representing a confused sense of longing."
    elif valence < 0.4 and energy < 0.4 and sentiment < -0.4:
        mood = 'Tangled Emotions'
        explanation = "The song has low valence, low energy, but the lyrics depict tangled and conflicting emotions."
    elif valence < 0.2 and energy > 0.6 and sentiment > 0.2:
        mood = 'Overwhelmed by Doubt'
        explanation = "The song has very low valence, high energy, and positive sentiment, evoking a feeling of being overwhelmed by doubt."
    elif valence < 0.2 and energy > 0.6 and sentiment < -0.2:
        mood = 'Confused Passion'
        explanation = "The song has very low valence, high energy, but the lyrics convey a confused and conflicted sense of passion."
    elif valence < 0.2 and energy < 0.4 and sentiment > 0.2:
        mood = 'Uncertain Desires'
        explanation = "The song has very low valence, low energy, and positive sentiment, reflecting uncertain and conflicting desires."
    elif valence < 0.2 and energy < 0.4 and sentiment < -0.2:
        mood = 'Lost in the Maze'
        explanation = "The song has very low valence, low energy, but the lyrics depict a state of being lost in a confusing maze of emotions."







# ...


    else:
        mood = 'Neutral or Indeterminate'
        explanation = "The song's audio features do not strongly align with a specific mood."

    statistics = f"Valence: {valence}\nEnergy: {energy}\nDanceability: {danceability}\nAcousticness: {acousticness}\nInstrumentalness: {instrumentalness}\nLoudness: {loudness}\nSpeechiness: {speechiness}\nTempo: {tempo}"

    return mood, explanation, statistics
