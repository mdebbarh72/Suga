import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.data.extracting import readData
import mlflow
import mlflow.sklearn

import src.utils.mlflowconf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluateCluster( df , k_range=[2,3,4,5,6,7,8,9,10,11]):

    mlflow.set_experiment('clustering')

    results = []

    for k in k_range:

        model = KMeans(n_clusters= k, n_init=10, random_state=42)

        with mlflow.start_run(run_name=f"kmeans k={k} new"):

            labels = model.fit_predict(df)
            inertia = model.inertia_
            silhouette = silhouette_score(df, labels)

            mlflow.log_params({
                'k' : k,
                'n_init' : 10,
                'random_state' : 42
            })

            mlflow.log_metrics({
                "silhouette" : silhouette,
                "inertia" : inertia
            })

            mlflow.sklearn.log_model(model,name='kmeans-diabetes-cluster')

            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette_score": silhouette
            })

    return pd.DataFrame( results )

def implemetnKmeans(df, k):

    model = KMeans(n_clusters=k, n_init=10, random_state=42)

    labels = model.fit_predict(df)

    return labels

def logPipeline(df, k):

    mlflow.set_experiment('kmeans final pipeline')

    with mlflow.start_run(run_name='kmeans-pipeline'):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=k, n_init=10, random_state=42))
        ])

        pipeline.fit(df)

        mlflow.sklearn.log_model(pipeline, name = 'kmeans pipeline')



def clusterResult(df, labels):

    diabetes_df = readData("clean")

    cluster_df = diabetes_df.copy()
    cluster_df["cluster"] = labels

    return cluster_df




