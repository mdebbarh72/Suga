import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow, mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import src.utils.mlflowconf

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

def getmodels():
    return {
        'Logistic': lr_pipeline,
        'knn':      knn_pipeline,
        'svm':      svm_pipeline,
    }





RANDOM_STATE = 42


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

param_grids = {

    "Logistic": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear", "lbfgs"]
    },

    "svm": {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]
    },

    "knn": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan", "minkowski"],
        "classifier__p": [1, 2]
    }
}

def hypertune(x_train, y_train):

    mlflow.set_experiment('hyperturning models')

    models = getmodels()

    tuning_results = []

    for name, model in models.items():
        with mlflow.start_run(run_name=f"hypertuning {name}"):
            print(f"\nTuning {name}...")


            search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring="f1_weighted",
                cv=cv,
                n_jobs=-1,
                refit=True
            )

            search.fit(x_train, y_train)
            best_params = search.best_params_
            best_score = search.best_score_
            best_estimator = search.best_estimator_

            mlflow.log_params({
                "best_params": best_params,
                "best_estimator": best_estimator
            })

            tuning_results.append({
                "name": name,
                "best_params": best_params,
                "best_score": best_score,
                "best_estimator": best_estimator
            })

            mlflow.log_metric('best score', best_score)

            print("Best parameters:")
            print(search.best_params_)

            print(f"Best CV F1: {search.best_score_:.4f}")


    tuning_summary = pd.DataFrame([
        {
            "model": result["name"],
            "best_params": result["best_params"],
            "best_score": result["best_score"]
        }
        for result in tuning_results
    ])

    return tuning_summary
