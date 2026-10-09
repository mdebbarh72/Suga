from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import src.data.extracting as extractor 
import src.data.cleaning as cleaner
import src.data.processing as prossessor

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger('prerocess')

def preprocess():

    #read data

    diabetes_df = extractor.readData()

    logger.info('passed: data extraction')

    #replace zero with nan
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_df = diabetes_df[cols].replace(0, np.nan)

    logger.info('passed: zero conversion to nan')

    #drop id column
    # diabetes_df = diabetes_df.drop(columns='ID')

    #impute data
    diabetes_df = cleaner.KNNImputation(diabetes_df)
    intCols = ['Glucose', 'BloodPressure', 'Insulin', 'Age']
    diabetes_df[intCols] = diabetes_df[intCols].round().astype('Int64')

    logger.info('passed: data imputation')
    #remove outliers
    diabetes_df = cleaner.deleteOutliers(diabetes_df)

    logger.info('passed: deleting outliers')

    #save clean data
    clean_data_path = project_root / "data" / "clean" / "clean_data.csv"
    clean_data_path.parent.mkdir(parents=True, exist_ok=True)
    diabetes_df.to_csv(clean_data_path, index=False) 

    logger.info('passed: save cleaned data')