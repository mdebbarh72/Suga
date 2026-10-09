import os

import joblib
import mlflow
import pandas as pd
import src.utils.mlflowconf

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.environ.get("MODEL_URI", "models:/diabetes-risk-classifier@production")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/diabetes_cluster_classifier.joblib")

mlflow.set_tracking_uri(MLFLOW_URI)

model = None


def loadModel():
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception:
        model = joblib.load(MODEL_PATH)
    return model


def predictRisk(patientData: dict):
    if model is None:
        raise RuntimeError("Model has not been loaded")

    data = pd.DataFrame([patientData])
    prediction = model.predict(data)[0]

    if prediction == 1:
        risk = "High"
    else:
        risk = "Normal"

    return risk
