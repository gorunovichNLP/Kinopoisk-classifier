# Локальный Kafka-контур

Kafka запускается одним broker/controller в KRaft-режиме. ZooKeeper не нужен.
Данные broker-а живут внутри dev-контейнера и сбрасываются при его пересоздании;
для интеграционных тестов это полезнее старого состояния между запусками.

## Почему два listener-а

- `localhost:9092` — адрес для Python-процессов на Windows;
- `kafka:19092` — адрес внутри Docker network.

Kafka-клиент использует bootstrap address только для первого соединения. Затем
broker возвращает `advertised.listeners`, поэтому адрес должен быть доступен из
той сети, где работает клиент.

## Запуск

```powershell
docker compose -f infra/docker/docker-compose.yml up -d kafka kafka-init
docker compose -f infra/docker/docker-compose.yml ps
```

`kafka-init` создаёт:

- `kinopoisk.reviews.v1` — 3 partitions;
- `kinopoisk.predictions.v1` — 3 partitions;
- `kinopoisk.reviews.dlq.v1` — 1 partition.

## Интеграционный тест

```powershell
$env:RUN_KAFKA_INTEGRATION="1"
python -m unittest tests.integration.test_kafka_worker -v
```

В тесте используется fake runtime. Это намеренно: сначала отдельно проверяем
Kafka, batching, публикацию результата и commit offsets, не смешивая возможные
ошибки брокера с загрузкой BERT из MinIO.

## Интеграционный тест с настоящей моделью

Когда быстрый Kafka-тест проходит, можно проверить весь inference-контур с
конкретной версией модели из MLflow Registry. Файлы этой версии скачиваются
через MLflow из MinIO; локальная папка с весами не используется.

```powershell
$env:RUN_REAL_MODEL_INTEGRATION="1"
$env:INFERENCE_MODEL_VERSION="2"
venv\Scripts\python -m unittest tests.integration.test_kafka_real_model -v
```

В примере `2` — не специальное значение: укажи номер нужной `READY`-версии из
своего Registry. В текущем локальном Registry корректный полный артефакт имеет
версию `2`.

Тест берёт один настоящий отзыв из `data/clean_dataset.csv`. Другой CSV с
колонкой `text` можно передать через `REAL_REVIEW_CSV`. Тест проверяет не
ожидаемую тональность конкретной фразы, а более важный контракт: сообщение
обработала реальная Registry-версия, а в output topic появился полностью
валидный `PredictionEventV1` с метаданными именно этой модели.
