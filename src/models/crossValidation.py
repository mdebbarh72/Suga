import sys 
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.dummy import DummyClassifier
from src.models.model_selection import getModels
from sklearn.metrics import make_scorer, precision_score

import mlflow, mlflow.sklearn
import src.utils.mlflowconf



RANDOM_STATE= 42

CV = StratifiedKFold(
    n_splits=5,
    shuffle= True,
    random_state=RANDOM_STATE
)

SCORING = {
    'accuracy' : 'accuracy',
    'balanced_accuracy' : "balanced_accuracy",
    "precision": make_scorer(
        precision_score,
        average="weighted",
        zero_division=0
    ),
    'recall' : 'recall_weighted',
    'f1_macro' : 'f1_macro',
    'f1_weighted' : 'f1_weighted'
}



def crossVlaidations(x_train, y_train):


    
    mlflow.set_experiment('Cross Validation')
    models = getModels()
    models['Dummy'] = DummyClassifier(
        strategy='most_frequent', 
        random_state=RANDOM_STATE
    )

    result = []

    for name, model in models.items():
        with mlflow.start_run(run_name=f'{name} cross_validation'):
            print("Started run:", mlflow.active_run().info.run_id)
            cv_result = cross_validate(
                model,
                x_train,
                y_train,
                cv = CV,
                scoring= SCORING,
                return_train_score=True,
                n_jobs=1
            )

            accuracy_mean = cv_result["test_accuracy"].mean()
            accuracy_std = cv_result["test_accuracy"].std()

            balanced_accuracy_mean = (
                cv_result["test_balanced_accuracy"].mean()
            )
            balanced_accuracy_std = (
                cv_result["test_balanced_accuracy"].std()
            )

            precision_mean = cv_result["test_precision"].mean()
            precision_std = cv_result["test_precision"].std()

            recall_mean = cv_result["test_recall"].mean()
            recall_std = cv_result["test_recall"].std()

            f1_weighted_mean = cv_result["test_f1_weighted"].mean()
            f1_weighted_std = cv_result["test_f1_weighted"].std()

            f1_macro_mean = cv_result["test_f1_macro"].mean()
            f1_macro_std = cv_result["test_f1_macro"].std()

            train_f1_weighted_mean = (
                cv_result["train_f1_weighted"].mean()
            )
            train_f1_weighted_std = (
                cv_result["train_f1_weighted"].std()
            )

            # Overfitting indicator
            f1_gap = (
                train_f1_weighted_mean
                - f1_weighted_mean
            )

            result.append({
                "model": name,

                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,

                "balanced_accuracy_mean": balanced_accuracy_mean,
                "balanced_accuracy_std": balanced_accuracy_std,

                "precision_mean": precision_mean,
                "precision_std": precision_std,

                "recall_mean": recall_mean,
                "recall_std": recall_std,

                "f1_weighted_mean": f1_weighted_mean,
                "f1_weighted_std": f1_weighted_std,

                "f1_macro_mean": f1_macro_mean,
                "f1_macro_std": f1_macro_std,

                "train_f1_weighted_mean": train_f1_weighted_mean,
                "train_f1_weighted_std": train_f1_weighted_std,

                "f1_gap": f1_gap
            })

            mlflow.log_param('name', name)

            mlflow.log_params({
                "cv_strategy": "StratifiedKFold",
                "n_splits": CV.n_splits,
                "shuffle": True,
                "random_state": RANDOM_STATE
            })


            mlflow.log_metrics({

                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,

                "balanced_accuracy_mean": balanced_accuracy_mean,
                "balanced_accuracy_std": balanced_accuracy_std,

                "precision_mean": precision_mean,
                "precision_std": precision_std,

                "recall_mean": recall_mean,
                "recall_std": recall_std,

                "f1_weighted_mean": f1_weighted_mean,
                "f1_weighted_std": f1_weighted_std,

                "f1_macro_mean": f1_macro_mean,
                "f1_macro_std": f1_macro_std,

                "train_f1_weighted_mean": train_f1_weighted_mean,
                "train_f1_weighted_std": train_f1_weighted_std,

                "f1_gap": f1_gap
            })



    return pd.DataFrame(result)
