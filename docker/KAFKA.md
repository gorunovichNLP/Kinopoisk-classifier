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
docker compose -f docker/docker-compose.yml up -d kafka kafka-init
docker compose -f docker/docker-compose.yml ps
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
