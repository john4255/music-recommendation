import chromadb
import text2emotion as emo
import pandas as pd
import os
import numpy as np

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("songs")

mp3_ids = []
genres = os.listdir('../data/MP3-Example')
genres = [g for g in genres if g != '.DS_Store']
for i, genre in enumerate(genres):
    songs = os.listdir(f'../data/MP3-Example/{genre}')
    songs = [s for s in songs if s != '.DS_Store']
    for j, song in enumerate(songs):
        track_id = song.split('.')[0].split('-')[1]
        mp3_ids.append(track_id)

listening_df = pd.read_csv('../data/User-Listening-History.csv')

listening_df = listening_df[listening_df['track_id'].isin(mp3_ids)]

users_test = listening_df.groupby('user_id').filter(lambda x: len(x) > 20)

n_success_agg = []
n_benchmark_agg = []
for user_id in users_test['user_id']:
    top_tracks = listening_df[listening_df['user_id'] == user_id].sort_values(by='playcount')['track_id'].unique()

    predictors = top_tracks[:3]
    vectors = []
    for track_id in predictors:
        result = collection.get(
            ids=[track_id],
            include=["documents", "embeddings"],
        )
        vectors.append(result['embeddings'][0])

    similar_docs = collection.query(
        query_embeddings=vectors,
        n_results=100,
    )
    sim_track_ids = similar_docs['ids'][0][3:]

    n_success = 0
    for sim_track_id in sim_track_ids:
        if sim_track_id in top_tracks:
            n_success += 1
    n_success_agg.append(n_success)

    n_benchmark = 0
    random_track_ids = listening_df.sample(n=100)
    for random_track_id in random_track_ids:
        if random_track_id in top_tracks:
            n_benchmark += 1
    n_benchmark_agg.append(n_benchmark)

print('=== RESULTS ===')
print(f'Mean Success = {np.mean(n_success_agg)}')
print(f'Std Success = {np.std(n_success_agg)}')
print(f'Mean Benchmark = {np.mean(n_benchmark_agg)}')
print(f'Std Benchmark = {np.std(n_benchmark_agg)}')