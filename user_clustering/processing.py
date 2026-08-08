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
def get_top_features(df, x=3, avg_col="tags"):

    # make sure there are 50+ streams for a user to be included
    total_streams = (df.groupby("user_id")["playcount"].sum().rename("total_streams"))
    valid_users = total_streams[total_streams >= 50].index
    df = df[df["user_id"].isin(valid_users)].copy()

    # keep only valid users in total_streams
    total_streams = total_streams.loc[valid_users]

    # helper to find the total x artists/genres/etc
    def get_top_x(df, feature):
        counts = (df.groupby(["user_id", feature], observed=True)["playcount"].sum().reset_index())

        counts["rank"] = (counts.groupby("user_id")["playcount"].rank(method="first", ascending=False))

        top = counts[counts["rank"] <= x].copy()
        top["rank"] = top["rank"].astype(int)

        return top

    # find most popular artists for individual user
    top_artists = get_top_x(df, "artist")

    # switch to wide
    artist_wide = top_artists.pivot(index="user_id", columns="rank", values="artist")

    artist_wide.columns = [f"artist_{i}" for i in artist_wide.columns]

    # repeat for genres
    top_genres = get_top_x(df, "genre")

    genre_wide = top_genres.pivot(index="user_id", columns="rank", values="genre")

    genre_wide.columns = [f"genre_{i}" for i in genre_wide.columns]

    # repeat for years
    top_years = get_top_x(df, "year")

    year_wide = top_years.pivot(index="user_id", columns="rank", values="year")

    year_wide.columns = [f"year_{i}" for i in year_wide.columns]

    # fix the tag formatting
    tag_df = df[["user_id", "tags", "playcount", "duration_ms", "danceability"]].dropna(subset=["tags"]).copy()

    # split comma-separated tags
    tag_df["tag"] = tag_df["tags"].str.split(",")

    # one tag per row
    tag_df = tag_df.explode("tag")

    # remove whitespace
    tag_df["tag"] = tag_df["tag"].str.strip()

    # aggregate statistics for each user/tag
    tag_stats = (
        tag_df.groupby(["user_id", "tag"], observed=True)
        .agg(
            playcount=("playcount", "sum"),
            avg_duration_ms=("duration_ms", "mean"),
            avg_danceability=("danceability", "mean")
        )
        .reset_index()
    )

    tag_stats["rank"] = (
        tag_stats.groupby("user_id")["playcount"]
        .rank(method="first", ascending=False)
    )

    top_tags = tag_stats[tag_stats["rank"] <= x].copy()
    top_tags["rank"] = top_tags["rank"].astype(int)

    tag_wide = top_tags.pivot(
        index="user_id",
        columns="rank",
        values="tag"
    )

    tag_wide.columns = [
        f"tag_{i}" for i in tag_wide.columns
    ]

    # find averages
    if avg_col == "tags":
        duration_wide = top_tags.pivot(
            index="user_id",
            columns="rank",
            values="avg_duration_ms"
        )

        duration_wide.columns = [
            f"tags_{i}_avg_duration_ms"
            for i in duration_wide.columns
        ]

        dance_wide = top_tags.pivot(
            index="user_id",
            columns="rank",
            values="avg_danceability"
        )

        dance_wide.columns = [
            f"tags_{i}_avg_danceability"
            for i in dance_wide.columns
        ]

    else:
        top_map = {
            "artist": top_artists,
            "genre": top_genres,
            "year": top_years
        }

        top_values = top_map[avg_col]

        # get averages for each user/feature combination
        averages = (
            df.groupby(["user_id", avg_col], observed=True)
            .agg(
                avg_duration_ms=("duration_ms", "mean"),
                avg_danceability=("danceability", "mean")
            )
            .reset_index()
        )

        # add averages to ranked top values
        top_values = top_values.merge(
            averages,
            on=["user_id", avg_col],
            how="left"
        )

        duration_wide = top_values.pivot(
            index="user_id",
            columns="rank",
            values="avg_duration_ms"
        )

        duration_wide.columns = [
            f"{avg_col}_{i}_avg_duration_ms"
            for i in duration_wide.columns
        ]

        dance_wide = top_values.pivot(
            index="user_id",
            columns="rank",
            values="avg_danceability"
        )

        dance_wide.columns = [
            f"{avg_col}_{i}_avg_danceability"
            for i in dance_wide.columns
        ]

    # combine into one df
    result = (
        total_streams.to_frame()
        .join(artist_wide, how="left")
        .join(genre_wide, how="left")
        .join(year_wide, how="left")
        .join(tag_wide, how="left")
        .join(duration_wide, how="left")
        .join(dance_wide, how="left")
        .reset_index()
    )

    return result


result_df = get_top_features(user_profile_df, x=1, avg_col="tags")

result_df.to_parquet("processed_user_data.parquet", index = False)