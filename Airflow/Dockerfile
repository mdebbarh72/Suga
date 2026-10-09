FROM apache/airflow:3.3.2

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY Airflow/requirements.txt /temp/requirements.txt

RUN pip install --no-cache-dir -r /temp/requirements.txt

COPY --chown=airflow:root src /opt/airflow/src
COPY --chown=airflow:root Airflow/dag.py /opt/airflow/dag.py

ENV PYTHONPATH=/opt/airflow