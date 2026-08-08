import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# load data
df = pd.read_parquet("processed_user_data.parquet")

output_df = df.copy()

# remove columns that shouldn't be clustered on
drop_cols = ["user_id", "total_streams"]
x = df.drop(columns = drop_cols)

# scale numeric features
numeric_cols = x.select_dtypes(include="number").columns

x[numeric_cols] = x[numeric_cols].fillna(0)

scaler = StandardScaler()

x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

# encode categoricals
x = pd.get_dummies(x, dtype = float)

x = x.fillna(0)

# dimensionality reduction
svd = TruncatedSVD(
    n_components = 15,
    random_state = 42
)

x_reduced = svd.fit_transform(x)

print("Explained variance: ", svd.explained_variance_ratio_.sum())

# train 3 cluster kmeans
model = KMeans(n_clusters = 3, random_state = 42, n_init = "auto")

clusters = model.fit_predict(x_reduced)

# apply cluster assignments to data
output_df["user_cluster"] = clusters

output_df.to_parquet("processed_data_w_clusters.parquet", index = False)

print("Saved cluster assignment data")

# interpret clusters

summary_rows = []

for cluster in sorted(output_df["user_cluster"].unique()):

    cluster_df = output_df[output_df["user_cluster"] == cluster]

    row = {
        "user_cluster": cluster,
        "num_users": len(cluster_df)
    }

    # most common categorical values
    categorical_cols = cluster_df.select_dtypes(include=["object", "string", "category"]).columns

    # remove user id
    categorical_cols = categorical_cols.drop(["user_id"], errors = "ignore")

    for col in categorical_cols:

        counts = cluster_df[col].value_counts()

        if len(counts) > 0:
            row[f"top_{col}"] = counts.index[0]
        else:
            row[f"top_{col}"] = None

    # average numeric features
    numeric_summary_cols = cluster_df.select_dtypes(include="number").columns

    numeric_summary_cols = numeric_summary_cols.drop(["user_cluster"], errors="ignore")

    # not 100% sure that this is necessary
    for col in numeric_summary_cols:
        # ignore user ids
        if col != "user_id":
            row[f"avg_{col}"] = cluster_df[col].mean()

    summary_rows.append(row)


cluster_summary = pd.DataFrame(summary_rows)

print("Cluster Summary:")
print(cluster_summary.to_string(index=False))

cluster_summary.to_csv("cluster_summary.csv", index=False)