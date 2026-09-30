import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = get_logger("etracting_data")


def KNNImputation(df):
    scaler = StandardScaler()
    imputer = KNNImputer(n_neighbors=3)

    scaled_data = scaler.fit_transform(df)
    imputed_data = imputer.fit_transform(scaled_data)

    imputed_data = scaler.inverse_transform(imputed_data)

    return pd.DataFrame(
        imputed_data,
        columns= df.columns,
        index = df.index
    )

    
def medianImputation(df):

    imputer = SimpleImputer(strategy='median')

    return pd.DataFrame(imputer.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

def iterativeImputation(df):

    imputer = IterativeImputer(max_iter=10, random_state=42)

    return pd.DataFrame(imputer.fit_transform(df), columns= df.columns, index= df.index)


def maskRows(df, percentage=0.10):

    df_masked = df.copy()
    masked_values = {}

    for col in df.columns:

        rows = df.index[df[col].notna()]
        n= int(len(rows) * percentage)
        selected = np.random.choice(rows, size=n, replace=False)
        df_masked.loc[selected, col] = np.nan

        masked_values = {
            "rows" : selected,
            "values" : df_masked.loc[selected, col].copy()
        }

    return df_masked, masked_values

def chooseImputeMethod(df):

    df_masked , masked_values = maskRows(df)

    df_knn = KNNImputation(df_masked)
    df_median = medianImputation(df_masked)
    df_iterative = iterativeImputation(df_masked)

    median_score = evaluateImputation(df_knn, masked_values)
    knn_score = evaluateImputation(df_knn, masked_values)
    iterative_score = evaluateImputation(df_iterative, masked_values)

    results = pd.DataFrame(
        {
            "method" : ['median', 'knn', 'iterative'],
            "MAE" : [median_score['MAE'], knn_score['MAE'], iterative_score['MAE']],
            "RMSE" : [median_score['RMSE'], knn_score['RMSE'], iterative_score['RMSE']]
        }
    )

    return results


def evaluateImputation(df_imputed, masked_values):
    rmse_scores = []
    mae_scores = []

    for col, data in masked_values.items():

        rows = data["rows"]
        original_values = data['values']
        predicted_values = df_imputed.loc[rows, col]

        mae = mean_absolute_error(
            original_values,
            predicted_values
        )

        rmse = mean_squared_error(
            original_values,
            predicted_values
        ) ** 0.5

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    return {
        "MAE" : mae_scores,
        "RMSE" : rmse_scores
    }



