import json
import joblib
import sklearn
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mlflow import MlflowClient

client = MlflowClient()

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


import src.utils.mlflowconf


def trainModel(
    summary,
    models,
    train_features,
    train_target,
    test_features,
    test_target
):


    best_row = summary.loc[summary["best_score"].idxmax()]

    best_model_name = best_row["model"]
    best_params = best_row["best_params"]

    print("Best model:", best_model_name)
    print("Best parameters:", best_params)


    final_pipeline = models[best_model_name].set_params(
        **best_params
    )

    mlflow.set_experiment('final training')


    with mlflow.start_run(run_name=f"final_{best_model_name}"):


        mlflow.log_param(
            "model",
            best_model_name
        )

        mlflow.log_param(
            "sklearn_version",
            sklearn.__version__
        )

        mlflow.log_param(
            "features",
            ", ".join(train_features.columns)
        )

        mlflow.log_param(
            "n_features",
            len(train_features.columns)
        )

        mlflow.log_param(
            "n_train_samples",
            len(train_features)
        )

        mlflow.log_param(
            "n_test_samples",
            len(test_features)
        )

    
        mlflow.log_param(
            "cluster_labels",
            ", ".join(
                map(
                    str,
                    sorted(train_target.unique().tolist())
                )
            )
        )

        # Log best hyperparameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(
                f"best_{param_name}",
                param_value
            )



        final_pipeline.fit(
            train_features,
            train_target
        )


        y_pred = final_pipeline.predict(
            test_features
        )


        cm = confusion_matrix(
            test_target,
            y_pred
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot(
            ax=ax,
            values_format="d"
        )

        ax.set_title(
            f"Confusion Matrix - {best_model_name}"
        )

        plt.tight_layout()


        confusion_matrix_path = PROJECT_ROOT / "models/confusion_matrix.png"

        plt.savefig(
            confusion_matrix_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


        # Log confusion matrix to MLflow
        mlflow.log_artifact(
            confusion_matrix_path,
            artifact_path="evaluation"
        )

        print(
            f"Confusion matrix saved to: "
            f"{confusion_matrix_path}"
        )


        test_metrics = {
            "accuracy": accuracy_score(
                test_target,
                y_pred
            ),

            "balanced_accuracy": balanced_accuracy_score(
                test_target,
                y_pred
            ),

            "precision_weighted": precision_score(
                test_target,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "recall_weighted": recall_score(
                test_target,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "f1_weighted": f1_score(
                test_target,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "f1_macro": f1_score(
                test_target,
                y_pred,
                average="macro",
                zero_division=0
            )
        }


        # Log all metrics
        mlflow.log_metrics(test_metrics)


        print("\nTest metrics:")

        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")



        model_path = PROJECT_ROOT / "models" / "diabetes_cluster_classifier.joblib"

        joblib.dump(
            final_pipeline,
            model_path
        )

        print(
            f"\nModel saved to: {model_path}"
        )


        # Log the joblib file as an artifact
        mlflow.log_artifact(
            model_path,
            artifact_path="joblib"
        )



        metadata = {
            "model_name": best_model_name,

            "features": list(
                train_features.columns
            ),

            "cluster_labels": sorted(
                train_target.unique().tolist()
            ),

            "sklearn_version": sklearn.__version__,

            "best_params": best_params,

            "test_metrics": test_metrics
        }


        metadata_path = PROJECT_ROOT / "models/diabetes_cluster_classifier_metadata.json"


        with open(
            metadata_path,
            "w"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )


        print(
            f"Metadata saved to: {metadata_path}"
        )


        # Log metadata as artifact
        mlflow.log_artifact(
            metadata_path,
            artifact_path="metadata"
        )



        features_path = PROJECT_ROOT / "models/features.txt"

        with open(
            features_path,
            "w"
        ) as file:

            file.write(
                "\n".join(
                    train_features.columns
                )
            )


        mlflow.log_artifact(
            features_path,
            artifact_path="metadata"
        )


        # mlflow.sklearn.log_model(
        #     final_pipeline,
        #     artifact_path="model"
        # )

        mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        name="diabetes-risk-classifier",
        registered_model_name="diabetes-risk-classifier"
    )



        model_name = "diabetes-risk-classifier"

        # Find the latest version
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        latest_version = max(
            versions,
            key=lambda v: int(v.version)
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=latest_version.version
        )

        print(
            f"Version {latest_version.version} is now production"
        )

        loaded_pipeline = joblib.load(
            model_path
        )


        original_predictions = (
            final_pipeline.predict(test_features)
        )

        loaded_predictions = (
            loaded_pipeline.predict(test_features)
        )


        predictions_match = np.array_equal(
            original_predictions,
            loaded_predictions
        )


        print(
            "Predictions match:",
            predictions_match
        )


        mlflow.log_param(
            "joblib_predictions_match",
            predictions_match
        )


        new_patient = pd.DataFrame([{
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }])


        prediction = loaded_pipeline.predict(
            new_patient
        )


        print(
            "Predicted cluster:",
            prediction[0]
        )


        mlflow.log_param(
            "example_patient_cluster",
            int(prediction[0])
        )



        print("\nMLflow run:")
        print("Run ID:", mlflow.active_run().info.run_id)
        print(
            "Experiment:",
            mlflow.get_experiment(
                mlflow.active_run().info.experiment_id
            ).name
        )