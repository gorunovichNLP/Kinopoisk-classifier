# MLflow-сервер с драйверами под Postgres (backend store) и S3/MinIO (artifacts).
# Голый mlflow-образ не содержит psycopg2 и boto3 — доставляем их сами.
#
# ВАЖНО: версию mlflow ЗАПИНЬ ту же, что в твоём venv (проверь: mlflow --version).
FROM python:3.11-slim

ARG MLFLOW_VERSION=3.15.1
RUN pip install --no-cache-dir \
    mlflow==${MLFLOW_VERSION} \
    psycopg2-binary \
    boto3

EXPOSE 5000