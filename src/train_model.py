import os
import sys
from pathlib import Path

sys.path.append("/opt/airflow/dags")

from Airflow.functions.preprocess_data import preprocess
from Airflow.functions.clustering import trainKmeans
from Airflow.functions.classification import classificationProcess

if __name__ == "__main__":
    print("Running preprocessing...")
    preprocess()
    print("Running clustering...")
    trainKmeans()
    print("Running classification...")
    classificationProcess()
    print("Done!")
