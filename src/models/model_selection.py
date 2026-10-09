from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

import time

import src.models.hypertuning as Optimizer
import mlflow, mlflow.sklearn

import src.utils.mlflowconf



def getModels():
    return {
        'logistic': make_pipeline(StandardScaler(), LogisticRegression()),
        'svm':      make_pipeline(StandardScaler(), SVC()),
        'knn':      make_pipeline(StandardScaler(), KNeighborsClassifier()),
    }


def trainAndEvaluate(name, model , x_train, y_train, x_test, y_test):
    start =  time.perf_counter()
    model = model.fit(x_train, y_train)
    pred  = model.predict(x_test)

    return {
        'name': name, 
        'model' : model,
        'accuracy' : accuracy_score(y_test, pred),
        'f1_weighted' : f1_score(y_test, pred, average='weighted', zero_division=0),
        'time': time.perf_counter() - start,
    }

def selectModel(summary, train_features, train_target, test_features):
    models = Optimizer.getmodels()

    # Select best model using CV weighted F1
    best_row = summary.loc[
        summary["best_score"].idxmax()
    ]

    best_model_name = best_row["model"]
    best_params = best_row["best_params"]

    print("Selected model:", best_model_name)
    print("CV weighted F1:", best_row["best_score"])
    print("Best parameters:", best_params)


    # Get the original pipeline
    best_model = models[best_model_name]

    # Apply the best hyperparameters
    best_model.set_params(**best_params)

    # Fit on ALL training data
    best_model.fit(train_features, train_target)


    y_pred = best_model.predict(test_features)

    return (best_row, best_model_name, best_model, best_params, y_pred)


def diagnostics(best_model, train_features, train_target, test_target, y_pred, best_row, best_model_name):

    mlflow.set_experiment('best model diagnostics')

    with mlflow.start_run(run_name='diagnostic run'):    # Train prediction
        y_train_pred = best_model.predict(train_features)

        # Scores
        train_f1 = f1_score(
            train_target,
            y_train_pred,
            average="weighted"
        )

        test_f1 = f1_score(
            test_target,
            y_pred,
            average="weighted"
        )

        # CV score from your summary
        cv_f1 = best_row["best_score"]

        print(f"Model: {best_model_name}")
        print(f"Train F1: {train_f1:.4f}")
        print(f"CV F1:    {cv_f1:.4f}")
        print(f"Test F1:  {test_f1:.4f}")

        mlflow.log_param('name', best_model_name)
        mlflow.log_params({
            'train_features': train_features.columns,
            'train_target' : train_target.name,
            'test_target' : test_target.name,
        })

        mlflow.log_metrics({
            'train f1' : train_f1,
            'cv f1' : cv_f1,
            'test_f1' : test_f1
        })

        mlflow.sklearn.log_model(best_model)