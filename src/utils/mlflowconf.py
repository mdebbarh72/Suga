
import os
import mlflow
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
