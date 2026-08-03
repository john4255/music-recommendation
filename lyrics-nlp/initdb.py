import whisper
import chromadb
import os
import text2emotion as emo

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
        name="songs",
        metadata={"hnsw:space": "cosine"}
    )
genres = os.listdir('../data/MP3-Example')
genres = [g for g in genres if g != '.DS_Store']
model = whisper.load_model('turbo')

for i, genre in enumerate(genres):
    songs = os.listdir(f'../data/MP3-Example/{genre}')
    songs = [s for s in songs if s != '.DS_Store']
    for j, song in enumerate(songs):
        print(f'Loading Genre {i+1}/{len(genres)} Song {j+1}/{len(songs)}...')
        path = f'../data/MP3-Example/{genre}/{song}'
        lyrics = model.transcribe(path)['text']

        print(f'Song: {song}')
        print(f'Lyrics: {lyrics}')

        track_id = song.split('.')[0].split('-')[1]
        emotions = emo.get_emotion(lyrics)
        top_emotions = sorted(emotions, key=emotions.get, reverse=True)[:2]

        print(track_id)

        # TODO: Check if lyrics is populated and check language
        collection.upsert(
                ids=[track_id],
                documents=[lyrics],
                metadatas=[{'Genre': genre,
                            'Top-Emotions': top_emotions,
                            **emotions}]
            )
