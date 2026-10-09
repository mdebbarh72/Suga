from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
sys.path.append("/opt/airflow/dags")
sys.path.append("/opt/airflow/src")
from functions.classification import classificationProcess
from functions.clustering import trainKmeans
from functions.preprocess_data import preprocess

default_args = {
    'owner' : 'neo flarux',
    'retries': 3,
    'retry_delay' : timedelta(minutes=15)
}


with DAG(
    dag_id='suga_dag',
    default_args= default_args,
    start_date= datetime(year=2026, month = 10, day=8),
    schedule='@weekly',
    catchup=False
) as dag:

    preprocessing = PythonOperator(
        task_id='preprocessing',
        python_callable=preprocess
    )

    clustering = PythonOperator(
        task_id='clustering',
        python_callable=trainKmeans
    )

    classification = PythonOperator(
        task_id='classification',
        python_callable=classificationProcess
    )

    preprocessing >> clustering >> classification