import pandas as pd
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# load data
def load_data(file):
    df = pd.read_parquet(file)

    df = df.sample(frac = 0.33, random_state = 42)

    # drop columns that we are not clustering on
    drop_cols = ["user_id", "total_streams"]

    x = df.drop(columns = drop_cols)

    # identiy numerics before one hot encoding
    numeric_cols = x.select_dtypes(include = "number").columns

    # fill missing numeric values
    x[numeric_cols] = x[numeric_cols].fillna(0)

    # scale numerics
    scaler = StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

    # convert categoricals to numeric
    x = pd.get_dummies(x, dtype = float)

    x = x.fillna(0)

    svd = TruncatedSVD(
        n_components = 20,
        random_state = 42
    )

    x = svd.fit_transform(x)

    print("Reduced shape: ", x.shape)
    print(
        "Explained variance:",
        svd.explained_variance_ratio_.sum()
    )

    return x 

# gaussian mixture model
def gmm(n, random_state = 42):
    model = GaussianMixture(
        n_components = n,
        covariance_type = "diag",
        random_state = random_state
    )

    return model

# bayesian gaussian mixture model
def bayesian_gmm(n, random_state = 42):
    model = BayesianGaussianMixture(
        n_components = n,
        covariance_type = "diag",
        random_state = random_state
    )

    return model

# k means
def kmeans(n, random_state = 42):
    model = KMeans(
        n_clusters = n,
        random_state = random_state,
        n_init = "auto"
    )

    return model

# function to evaluate model
def evaluate_model(model, x, model_name):

    # train model
    model.fit(x)

    # cluster assignments
    labels = model.predict(x)

    # common clustering metrics
    silhouette = silhouette_score(x, labels)
    davies_bouldin = davies_bouldin_score(x, labels)

    results = {
        "model": model_name,
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin
    }

    # gmm specific metrics
    if isinstance(model, GaussianMixture):
        results["aic"] = model.aic(x)
        results["bic"] = model.bic(x)
    
    else:
        results["aic"] = None
        results["bic"] = None

    return results

if __name__ == "__main__":

    x = load_data("processed_user_data.parquet")

    print(x.shape)

    results = []

    # clusters
    for n in range(2, 8):

        models = [
            ("GMM", gmm(n)),
            ("Bayesian GMM", bayesian_gmm(n)),
            ("K-Means", kmeans(n))
        ]

        for model_name, model in models:
            print(f"Training {model_name} for n = {n}")

            metrics = evaluate_model(model, x, model_name)

            metrics["n_clusters"] = n

            results.append(metrics)

    # convert to a df
    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "model",
            "n_clusters",
            "silhouette_score",
            "davies_bouldin_score",
            "aic",
            "bic"
        ]
    ]

    print("\nModel Performance:")
    print(results_df)