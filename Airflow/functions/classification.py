from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np

import src.data.extracting as Extractor
import src.models.crossValidation as Validator
import src.models.model_selection as Selector
import src.models.hypertuning as Tuner
import src.models.finalTraining as Trainer
from sklearn.model_selection import train_test_split
import src.models.baseline_training as BaseTrain


RANDOM_STATE = 42


def classificationProcess():
    cluster_df = Extractor.readData('cluster')

    features = cluster_df.drop(columns=['cluster'])
    target = cluster_df['cluster']

    train_features , test_features, train_target, test_target = train_test_split(features, target, test_size = 0.3, stratify = target, random_state = RANDOM_STATE )

    result = BaseTrain.train(train_features, train_target, test_features, test_target)

    cv_result = Validator.crossVlaidations(train_features, train_target)

    summary = Tuner.hypertune(train_features, train_target)

    best_row, best_model_name, best_model, best_params, y_pred= Selector.selectModel(summary, train_features, train_target, test_features)

    Selector.diagnostics(best_model, train_features, train_target, test_target, y_pred, best_row, best_model_name)

    models= Tuner.getmodels()

    Trainer.trainModel(summary, models, train_features, train_target, test_features, test_target)