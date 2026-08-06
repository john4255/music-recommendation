import chromadb
import os
from msclap import CLAP

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
        name="songs_clap",
        metadata={"hnsw:space": "cosine"}
    )

clap_model = CLAP(version='2023', use_cuda=False)

genres = os.listdir('../data/MP3-Example')
genres = [g for g in genres if g != '.DS_Store']

for i, genre in enumerate(genres):
    songs = os.listdir(f'../data/MP3-Example/{genre}')
    songs = [s for s in songs if s != '.DS_Store']
    for j, song in enumerate(songs):
        print(f'Loading Genre {i+1}/{len(genres)} Song {j+1}/{len(songs)}...')

        track_id = song.split('.')[0].split('-')[1]
        path = f'../data/MP3-Example/{genre}/{song}'

        audio_embeddings = clap_model.get_audio_embeddings([path])
        collection.upsert(
                ids=[track_id],
                documents=['placeholder'],
                embeddings=[audio_embeddings[0].numpy()],
            )