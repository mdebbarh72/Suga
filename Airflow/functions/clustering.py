from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

import src.models.kmeans as ClusterModel
import src.data.extracting as Extractor
from src.utils.logger import get_logger

logger = get_logger('clustering')

def trainKmeans():

    diabetes_df = Extractor.readData('clean')
    logger.info('passed: loading data')

    scores = ClusterModel.evaluateCluster(diabetes_df)
    logger.info('passed: testing different k')

    best_k = scores.loc[scores["silhouette_score"].idxmax(), "k"]
    logger.info('passed : selecting the best k')

    labels = ClusterModel.implemetnKmeans(diabetes_df, best_k)
    logger.info('passed : labeling clusters')


    clustered_df = ClusterModel.clusterResult(diabetes_df, labels)
    logger.info('passed: adding labels to the data')

    clustered_df.to_csv('../../data/processed/cluster_data.csv', index= False)
    logger.info('passed: saved data')

    
