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

def standardizeTrainingData(train, test):

    scaler = StandardScaler()

    fitted_data = scaler.fit(train)

    scaled_train = scaler.transform(train)
    scaled_test= scaler.transform(test)

    return (pd.DataFrame(scaled_train, columns=train.columns, index = train.index),
             pd.DataFrame(scaled_test, columns= test.columns, index = test.index))
