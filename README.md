# Kinopoisk Classifier

## Project structure

- `kinopoisk_classifier/` contains the production services and shared train/serve logic;
- `training/` contains model training, calibration, and MLflow registration;
- `experiments/` contains ensemble experiments and external baselines;
- `contracts/` contains MongoDB and PostgreSQL schemas;
- `infra/docker/` contains the local Kafka, MongoDB, MLflow, MinIO, and PostgreSQL stack;
- `tests/` contains integration and end-to-end tests.

## Create a virtual environment

```powershell
python -m venv venv
```

## Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

## Start the local infrastructure

```powershell
docker compose -f infra\docker\docker-compose.yml up -d
```

## Run the MongoDB Review Producer

After MongoDB and Kafka are running:

```powershell
python -m kinopoisk_classifier.review_producer
```

Override settings with environment variables prefixed with
`REVIEW_PRODUCER_`, for example:

```powershell
$env:REVIEW_PRODUCER_BATCH_SIZE="50"
$env:REVIEW_PRODUCER_POLL_INTERVAL_SECONDS="1"
python -m kinopoisk_classifier.review_producer
```

Stop the service with `Ctrl+C`.

## Run the Kafka Prediction Writer

After Kafka and `predictions-postgres` are running:

```powershell
python -m kinopoisk_classifier.prediction_writer
```

Override settings with environment variables prefixed with
`PREDICTION_WRITER_`. Stop the service with `Ctrl+C`.

## Run the real-model integration test

```powershell
$env:RUN_REAL_MODEL_INTEGRATION="1"
$env:INFERENCE_MODEL_VERSION="2"

venv\Scripts\python -m unittest tests.integration.test_kafka_real_model -v
```
