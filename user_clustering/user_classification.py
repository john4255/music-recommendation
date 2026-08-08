import kagglehub
import pandas as pd
import os

# download data or used cache copy
path = kagglehub.dataset_download("undefinenull/million-song-dataset-spotify-lastfm")

# create dfs for music data and user id data
music_df = pd.read_csv(os.path.join(path, "Music Info.csv"))
user_df = pd.read_csv(os.path.join(path, "User Listening History.csv"))

# select only features of interest
music_cols = music_df[["track_id", "artist", "tags", "genre", "year", "duration_ms", "danceability"]]
user_cols = user_df[["user_id", "track_id", "playcount"]]

# join on track_id, there will be rows with duplicate user_ids
user_profile_df = pd.merge(music_cols, user_cols, on = "track_id")

# for each unique user_id, rank the top x number of artists, tags, genres, years
# for the top 3 tags, get the average duration_ms and danceability
# avg_col must be one of: "artist", "genre", "year", "tags"
def get_top_features(user_profile_df, x, avg_col):
    results = []

    for user_id, group in user_profile_df.groupby("user_id"):
        row = {"user_id": user_id}

        # top artists
        artist_counts = group.groupby("artist")["playcount"].sum().sort_values(ascending=False)
        top_artists = artist_counts.head(x).index.tolist()
        for each in range(x):
            row[f"artist_{each + 1}"] = top_artists[each] if each < len(top_artists) else None

        # top genres 
        genre_counts = group.groupby("genre")["playcount"].sum().sort_values(ascending=False)
        top_genres = genre_counts.head(x).index.tolist()
        for each in range(x):
            row[f"genre_{each + 1}"] = top_genres[each] if each < len(top_genres) else None

        # top years
        year_counts = group.groupby("year")["playcount"].sum().sort_values(ascending=False)
        top_years = year_counts.head(x).index.tolist()
        for each in range(x):
            row[f"year_{each + 1}"] = top_years[each] if each < len(top_years) else None

        # top tags, split the comma-separated mess
        tag_rows = []
        for each, r in group.iterrows():
            if pd.isna(r["tags"]):
                continue
            for t in [tag.strip() for tag in r["tags"].split(",")]:
                tag_rows.append({"tag": t,
                                "playcount": r["playcount"],
                                "duration_ms": r["duration_ms"],
                                "danceability": r["danceability"]
                                })
                
        tag_df = pd.DataFrame(tag_rows)

        if not tag_df.empty:
            tag_counts = tag_df.groupby("tag")["playcount"].sum().sort_values(ascending=False)
            top_tags = tag_counts.head(x).index.tolist()
        else:
            top_tags = []

        for each in range(x):
            row[f"tag_{each + 1}"] = top_tags[each] if each < len(top_tags) else None

        # averages for avg_col 
        if avg_col == "tags":
            for each in range(x):
                tag = top_tags[each] if each < len(top_tags) else None
                if tag is not None:
                    subset = tag_df[tag_df["tag"] == tag]
                    row[f"{avg_col}_{each + 1}_avg_duration_ms"] = subset["duration_ms"].mean()
                    row[f"{avg_col}_{each + 1}_avg_danceability"] = subset["danceability"].mean()
                else:
                    row[f"{avg_col}_{each + 1}_avg_duration_ms"] = None
                    row[f"{avg_col}_{each + 1}_avg_danceability"] = None
        else:
            top_vals_map = {"artist": top_artists, "genre": top_genres, "year": top_years}
            top_vals = top_vals_map[avg_col]
            for each in range(x):
                val = top_vals[each] if each < len(top_vals) else None
                if val is not None:
                    subset = group[group[avg_col] == val]
                    row[f"{avg_col}_{each + 1}_avg_duration_ms"] = subset["duration_ms"].mean()
                    row[f"{avg_col}_{each + 1}_avg_danceability"] = subset["danceability"].mean()
                else:
                    row[f"{avg_col}_{each + 1}_avg_duration_ms"] = None
                    row[f"{avg_col}_{each + 1}_avg_danceability"] = None

        results.append(row)

    return pd.DataFrame(results)

result_df = get_top_features(user_profile_df, x=3, avg_col="tags")

print(result_df.head(5))