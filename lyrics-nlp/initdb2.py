import whisper
import chromadb
import os
import text2emotion as emo

client = chromadb.PersistentClient(path="./chroma_db")

collection1 = client.get_or_create_collection(
        name="songs",
        metadata={"hnsw:space": "cosine"}
    )
collection = client.get_or_create_collection(
        name="songs_emotions",
        metadata={"hnsw:space": "cosine"}
    )

emotion_labels = ['Angry', 'Surprise', 'Sad', 'Fear', 'Happy']
all_records = collection1.get()
track_ids = all_records['ids']

for i, track_id in enumerate(track_ids):
    document = all_records['documents'][i]
    metadata = all_records['metadatas'][i]

    collection.upsert(
                ids=[track_id],
                documents=[document],
                embeddings=[metadata[label] for label in emotion_labels],
            )