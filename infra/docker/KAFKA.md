# Local Kafka environment

Kafka runs as a single broker/controller in KRaft mode and does not require
ZooKeeper. Broker data lives inside the development container and is reset when
that container is recreated. This keeps integration tests isolated from stale
state.

## Why two listeners are required

- `localhost:9092` is used by Python processes running on Windows;
- `kafka:19092` is used inside the Docker network.

A Kafka client uses the bootstrap address only for its initial connection. The
broker then returns `advertised.listeners`, so the advertised address must be
reachable from the client's network.

## Start Kafka

```powershell
docker compose -f infra/docker/docker-compose.yml up -d kafka kafka-init
docker compose -f infra/docker/docker-compose.yml ps
```

`kafka-init` creates:

- `kinopoisk.reviews.v1` with 3 partitions;
- `kinopoisk.predictions.v1` with 3 partitions;
- `kinopoisk.reviews.dlq.v1` with 1 partition.

## Integration test

```powershell
$env:RUN_KAFKA_INTEGRATION="1"
python -m unittest tests.integration.test_kafka_worker -v
```

This test uses a fake runtime so that Kafka, batching, result publication, and
offset commits can be verified independently from BERT and MinIO failures.

## Integration test with a real model

After the fast Kafka test passes, verify the inference path with a specific
MLflow Registry model version. MLflow downloads that version from MinIO; the
test does not use a local weights directory.

```powershell
$env:RUN_REAL_MODEL_INTEGRATION="1"
$env:INFERENCE_MODEL_VERSION="2"
venv\Scripts\python -m unittest tests.integration.test_kafka_real_model -v
```

The value `2` is only an example. Set the number of the required `READY` version
from your registry. In the current local registry, version `2` contains a
complete serving artifact.

The test reads one real review from `data/clean_dataset.csv`. Set
`REAL_REVIEW_CSV` to use another CSV containing a `text` column. The assertion
does not depend on the expected sentiment of one phrase. It verifies that the
configured registry version processed the event and produced a fully valid
`PredictionEventV1` with matching model metadata.
