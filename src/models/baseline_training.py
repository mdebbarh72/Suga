
import src.models.model_selection as Selector
import time
import mlflow
import mlflow.sklearn

models =  Selector.getModels()

import mlflow
from pathlib import Path

import pandas as pd



def train(train_features, train_target, test_features, test_target):
    mlflow.set_experiment('baseline training')


    t0 = time.perf_counter()
    results = []
    for name , model in models.items():
        with mlflow.start_run(run_name=f'{name} baseline'):
            result = Selector.trainAndEvaluate(name, model, train_features, train_target, test_features, test_target)
            results.append(result)
            mlflow.log_param('name',model)

            mlflow.log_metrics({
                'accuracy': result['accuracy'],
                'f1_score' : result['f1 score'],
                'time' : result['time']
            })

            mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")

        
    full_time = time.perf_counter() - t0

    print(f'fulltime: {full_time}')

    results = pd.DataFrame(results)

    return results


