# Kinopoisk Classifier

## Структура проекта

- `kinopoisk_classifier/` — production-код inference-сервиса и общая train/serve-логика;
- `training/` — обучение, калибровка и загрузка модели в MLflow/MinIO;
- `experiments/` — ensemble-эксперимент и внешняя baseline-модель;
- `contracts/` — схемы Kafka, MongoDB и PostgreSQL;
- `infra/docker/` — локальные Kafka, MongoDB, MLflow, MinIO и PostgreSQL;
- `tests/` — интеграционные тесты.

## Создать окружение

```powershell
python -m venv venv
```

## Активировать окружение

```powershell
.\venv\Scripts\Activate.ps1
```

## Поднять локальную инфраструктуру

```powershell
docker compose -f infra\docker\docker-compose.yml up -d
```

## Запустить MongoDB Review Producer

После запуска MongoDB и Kafka:

```powershell
python -m kinopoisk_classifier.review_producer
```

Настройки можно переопределять переменными окружения с префиксом
`REVIEW_PRODUCER_`, например:

```powershell
$env:REVIEW_PRODUCER_BATCH_SIZE="50"
$env:REVIEW_PRODUCER_POLL_INTERVAL_SECONDS="1"
python -m kinopoisk_classifier.review_producer
```

Остановить сервис можно сочетанием `Ctrl+C`.

## Запустить Kafka Prediction Writer

После запуска Kafka и `predictions-postgres`:

```powershell
python -m kinopoisk_classifier.prediction_writer
```

Настройки можно переопределять переменными с префиксом
`PREDICTION_WRITER_`. Остановить сервис можно сочетанием `Ctrl+C`.

## Запустить тест с реальной моделью

```powershell
$env:RUN_REAL_MODEL_INTEGRATION="1"
$env:INFERENCE_MODEL_VERSION="2"

venv\Scripts\python -m unittest tests.integration.test_kafka_real_model -v
```
