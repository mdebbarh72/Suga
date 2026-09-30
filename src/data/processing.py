import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



def standardize(df):

    scaler = StandardScaler()
    scaled_data= scaler.fit_transform(df)

    return pd.DataFrame(
        scaled_data,
        columns= df.columns,
        index= df.index
    )

def normalize(df):

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    return pd.DataFrame(
        scaled_data,
        columns = df.columns,
        index= df.index
    )

def robustScale(df):

    scaler= RobustScaler()
    scaled_data = scaler.fit_transform(df)

    return pd.DataFrame(
        scaled_data,
        columns=df.columns,
        index= df.index
    )

