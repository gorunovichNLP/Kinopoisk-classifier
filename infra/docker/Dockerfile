# MLflow server with PostgreSQL backend-store and S3/MinIO artifact drivers.
# The base Python image does not include these dependencies.
#
# Keep this version aligned with the project virtual environment.
FROM python:3.11-slim

ARG MLFLOW_VERSION=3.15.1
RUN pip install --no-cache-dir \
    mlflow==${MLFLOW_VERSION} \
    psycopg2-binary \
    boto3

EXPOSE 5000
