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

## Запустить тест с реальной моделью

```powershell
$env:RUN_REAL_MODEL_INTEGRATION="1"
$env:INFERENCE_MODEL_VERSION="2"

venv\Scripts\python -m unittest tests.integration.test_kafka_real_model -v
```
