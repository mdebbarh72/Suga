FROM python:3.12-slim

WORKDIR /mlflow

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow==3.16.1