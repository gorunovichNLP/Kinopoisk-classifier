# Kinopoisk Classifier: Implementation History and Video Plan

This document presents the project in the order in which it can be explained in
an educational video, from the business objective and offline metric to the
production inference pipeline. Application containerization and deployment to
Selectel are planned but not implemented yet.

## Final system overview

The project classifies Russian movie reviews as `neg`, `neu`, or `pos`. Its
production path contains three application services:

1. `ReviewProducer` reads new MongoDB reviews and publishes them to Kafka.
2. `InferenceWorker` loads one immutable RuBERT version from MLflow/MinIO,
   consumes reviews, and publishes predictions.
3. `PredictionWriter` consumes predictions and stores their history in
   PostgreSQL with idempotent writes.

```text
MongoDB reviews
    -> ReviewProducer
    -> Kafka: kinopoisk.reviews.v1
    -> InferenceWorker <- MLflow Registry / MinIO
    -> Kafka: kinopoisk.predictions.v1
    -> PredictionWriter
    -> PostgreSQL: sentiment.prediction_history
```

Invalid inference inputs are retained in `kinopoisk.reviews.dlq.v1`.

## Implementation chronology

### 1. Define the business problem and metric

The business objective is to automate review sentiment classification so that
analytics or moderation teams do not need to inspect every review manually.
Macro-F1 is the primary offline ML metric, not the business metric. Suitable
business outcomes include automated coverage, processing cost per 1,000
reviews, time to insight, and the share of decisions later corrected by a
human.

Relevant files:

- `training/metrics.py` computes macro-F1 and diagnostic metrics;
- `training/train.py` selects the best checkpoint by macro-F1;
- a future `docs/product-metrics.md` should formalize business KPIs and online
  guardrails.

### 2. Download the dataset and perform EDA

The Kaggle dataset is loaded with KaggleHub. After removing 96 blank or
duplicate records, 131,573 reviews remain: 87,091 positive, 24,678 neutral, and
19,804 negative. EDA also shows that 36.2% of reviews exceed BERT's 512-token
limit.

Relevant file: `training/notebooks/EDA.ipynb`.

### 3. Fix the label and encoding contracts

The canonical order is `neg -> 0`, `neu -> 1`, `pos -> 2`. The base model is
`DeepPavlov/rubert-base-cased`, the maximum input length is 512, and long inputs
keep 256 head tokens and 254 tail tokens plus special tokens.

Relevant file: `kinopoisk_classifier/shared/contracts.py`.

### 4. Build reproducible data preparation

Dataset loading, snapshots, cleaning, and a stratified 70/15/15 split with
`seed=42` were moved from the notebook into reusable Python code. Duplicates are
removed before splitting to avoid leakage, and stratification protects the
minority neutral class.

Relevant file: `training/data.py`.

### 5. Share head-and-tail encoding between training and serving

Long reviews keep their beginning and conclusion because the final verdict is
often located near the end. One shared implementation prevents train/serve
skew, while dynamic batch padding reduces unnecessary computation.

Relevant files:

- `kinopoisk_classifier/shared/encoding.py`;
- `kinopoisk_classifier/inference/runtime.py`.

### 6. Fine-tune RuBERT for three-class sentiment classification

RuBERT receives a new three-class classification head. Weighted cross-entropy
addresses class imbalance. The pretrained encoder uses learning rate `2e-5`,
while the randomly initialized head uses `1e-4`. The best checkpoint is chosen
by validation macro-F1 and evaluated once on the test split.

Relevant files:

- `training/model.py`;
- `training/train.py`;
- `training/metrics.py`.

### 7. Add experiment tracking and artifact storage

MLflow records parameters, metrics, runs, and model versions. MLflow metadata
is stored in a dedicated PostgreSQL database, while model artifacts are stored
in MinIO. The existing `infra/docker/Dockerfile` builds only the MLflow server;
it is not an application-service image.

Relevant files:

- `infra/docker/docker-compose.yml`;
- `infra/docker/Dockerfile`;
- `training/train.py`;
- `training/upload_model.py`.

### 8. Calibrate confidence and select a decision strategy

The project compares calibrated argmax, per-class logit bias, and per-class
probability thresholds. Parameters are selected on validation data and tested
once on the test split. The stored GPU result selects `class_bias`, temperature
`1.47061026096344`, and bias `[-1.5, -0.25, 0.0]`. Test macro-F1 is `0.7525567`
versus `0.7525296` for uncalibrated argmax.

Relevant files:

- `training/thresholds.py`;
- `kinopoisk_classifier/shared/decision.py`;
- `artifacts_from_gpu/thresholds_from_gpu.json`.

### 9. Evaluate an ensemble

The project model is compared with
`seara/rubert-base-cased-russian-sentiment`. The adapter aligns the external
model's labels and input length with the project contract. The best ensemble
uses weight `0.8` for the project model and reaches test macro-F1 `0.7529634`,
only `0.0004339` above the standalone model. The production runtime therefore
keeps one model instead of doubling inference cost.

Relevant files:

- `experiments/baselines/seara/runtime.py`;
- `experiments/ensemble/evaluate.py`;
- `experiments/ensemble/results.json`.

### 10. Package an immutable serving artifact

One Model Registry version contains weights, Transformers configuration,
tokenizer, `label_map.json`, and `thresholds.json`. The runtime requires an
explicit positive version number, verifies the artifact before loading its
weights, and rejects incompatible label mappings or model configurations.

Relevant files:

- `training/upload_model.py`;
- `kinopoisk_classifier/inference/runtime.py`;
- `kinopoisk_classifier/shared/schemas.py`.

### 11. Define versioned data contracts

Strict Pydantic models validate MongoDB reviews, Kafka review events, prediction
events, dead-letter records, model metadata, probabilities, and identifiers.
A deterministic SHA-256 prediction identifier enables idempotent downstream
writes. MongoDB and PostgreSQL schemas enforce additional storage constraints.

Relevant files:

- `kinopoisk_classifier/shared/schemas.py`;
- `contracts/mongodb/review-v1.schema.json`;
- `contracts/postgresql/001_create_prediction_history.sql`.

### 12. Create the local Kafka environment

Kafka runs as one broker/controller in KRaft mode. Separate internal and host
listeners support Docker and Windows clients. A one-shot init container creates
the review, prediction, and dead-letter topics.

Relevant files:

- `infra/docker/docker-compose.yml`;
- `infra/docker/KAFKA.md`;
- `tests/integration/test_kafka_worker.py`;
- `tests/integration/test_kafka_real_model.py`.

### 13. Implement InferenceWorker

The worker consumes and validates review events in batches, invokes
`ModelRuntime`, publishes prediction events, and redirects invalid input to the
dead-letter topic. Automatic offset commits are disabled. Input offsets are
committed only after Kafka acknowledges every output record, providing
at-least-once processing.

Relevant files:

- `kinopoisk_classifier/inference/config.py`;
- `kinopoisk_classifier/inference/runtime.py`;
- `kinopoisk_classifier/inference/worker.py`.

### 14. Refactor the repository into packages

Training, experiments, production services, shared logic, contracts,
infrastructure, and tests were separated. Shared encoding and decision code
ensure that experiments and production use the same behavior.

Relevant directories: `training/`, `experiments/`, `kinopoisk_classifier/`,
`contracts/`, `infra/`, and `tests/`.

### 15. Add MongoDB as the immutable review source

MongoDB was added to the local stack. Reviews are read in ascending ObjectId
order using `_id > last_review_id`, which allows efficient continuation after a
restart.

Relevant files:

- `kinopoisk_classifier/review_producer/config.py`;
- `kinopoisk_classifier/review_producer/mongo.py`;
- `contracts/mongodb/review-v1.schema.json`.

### 16. Implement ReviewProducer

ReviewProducer coordinates an ordered MongoDB reader, Kafka publisher, and
persistent checkpoint store. Its critical sequence is `load checkpoint -> read
batch -> publish -> save checkpoint`. The checkpoint advances only after Kafka
acknowledges the complete batch and cannot move backwards.

Relevant files:

- `kinopoisk_classifier/review_producer/mongo.py`;
- `kinopoisk_classifier/review_producer/kafka.py`;
- `kinopoisk_classifier/review_producer/checkpoint.py`;
- `kinopoisk_classifier/review_producer/producer.py`;
- `kinopoisk_classifier/review_producer/__main__.py`.

### 17. Add a dedicated prediction-history database

`predictions-postgres` is separate from the MLflow database. Its migration
creates prediction history, indexes, validation constraints, and a view of the
latest prediction for each review.

Relevant files:

- `contracts/postgresql/001_create_prediction_history.sql`;
- `infra/docker/docker-compose.yml`.

### 18. Implement PredictionWriter

The writer validates Kafka prediction events, persists them transactionally,
and commits Kafka offsets only after PostgreSQL commit. `INSERT ... ON CONFLICT
DO NOTHING` turns repeated at-least-once delivery into a safe no-op.

Relevant files:

- `kinopoisk_classifier/prediction_writer/config.py`;
- `kinopoisk_classifier/prediction_writer/postgres.py`;
- `kinopoisk_classifier/prediction_writer/writer.py`;
- `kinopoisk_classifier/prediction_writer/__main__.py`.

### 19. Verify the complete pipeline

The end-to-end integration test inserts a real review into MongoDB and follows
it through ReviewProducer, Kafka, a real model loaded from MLflow/MinIO,
InferenceWorker, PredictionWriter, and PostgreSQL. It verifies the checkpoint,
event identity, exact model version, and final database row.

Relevant file: `tests/integration/test_full_inference_pipeline.py`.

## Planned work

### 20. Containerize the three application services

Create one initial application image and run it with three different commands.
Add `.dockerignore`, keep secrets outside the image, use Docker DNS names for
infrastructure services, set an immutable image tag, and verify the complete
containerized pipeline. A later optimization can separate the heavy inference
dependencies from the lightweight I/O services.

Planned files:

- `infra/docker/Dockerfile.services`;
- `.dockerignore`;
- `infra/docker/docker-compose.services.yml`;
- `infra/docker/.env.services.example`.

### 21. Deploy to Selectel

The first educational deployment can use a Containers Ready cloud server,
Selectel Container Registry, and Docker Compose. A self-contained demonstration
must deploy not only the three application containers but also their MongoDB,
Kafka, PostgreSQL, MLflow, and object-storage dependencies. A production-grade
version should isolate stateful services, restrict them to a private network,
store secrets outside Git, back up persistent data, collect logs and metrics,
and support rollback to immutable image and model versions.

Planned files:

- `infra/selectel/docker-compose.yml`;
- `infra/selectel/.env.example`;
- `infra/selectel/README.md`;
- optionally `infra/selectel/cloud-init.yaml` or Terraform configuration.

The deployment acceptance criterion is a new MongoDB review reaching
PostgreSQL through all three continuously running containers with a valid model
version, while restarts preserve checkpoints, offsets, and idempotency.

## Suggested video chapters

1. Business objective and macro-F1.
2. EDA, imbalance, and long reviews.
3. Shared preprocessing and RuBERT training.
4. MLflow, calibration, and the ensemble result.
5. Immutable model artifacts and data contracts.
6. Kafka and InferenceWorker.
7. MongoDB and ReviewProducer checkpoints.
8. PostgreSQL and idempotent PredictionWriter writes.
9. End-to-end integration testing.
10. Application containerization.
11. Selectel deployment and final demonstration.

The central lesson is that a trained model is only one component of a reliable
ML system. Reproducible data, immutable artifacts, versioned contracts,
checkpoints, offsets, dead-letter handling, idempotency, tests, and repeatable
deployment are equally important.
