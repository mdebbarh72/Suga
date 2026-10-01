import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.data.extracting import readData

def evaluateCluster( df , k_range=[2,3,5,7,9,10,11]):

    results = []

    for k in k_range:

        model = KMeans(n_clusters= k, n_init=10, random_state=42)

        labels = model.fit_predict(df)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(df, labels)
        })

    return pd.DataFrame( results )

def implemetnKmeans(df, k):

    model = KMeans(n_clusters=k, n_init=10, random_state=42)

    labels = model.fit_predict(df)

    return labels



def clusterResult(df, labels):

    diabetes_df = readData("cleaned")

    cluster_df = diabetes_df.copy()
    cluster_df["cluster"] = labels

    return cluster_df




