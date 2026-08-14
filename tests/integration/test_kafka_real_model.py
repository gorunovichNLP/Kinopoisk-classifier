r"""Полный интеграционный тест inference-контура с настоящей моделью.

Проверяем один реальный отзыв по маршруту:

    dataset -> Kafka reviews -> InferenceWorker -> MLflow/MinIO model
            -> Kafka predictions -> PredictionEventV1

Mongo Producer здесь пока намеренно не участвует: его добавим отдельным шагом.
Сейчас важно проверить границу Kafka + настоящий runtime, чтобы при ошибке было
понятно, в какой части контура её искать.

Тест тяжёлый и поэтому выключен по умолчанию. Запуск из корня проекта:

    $env:RUN_REAL_MODEL_INTEGRATION="1"
    venv\Scripts\python -m unittest tests.integration.test_kafka_real_model -v
"""

import csv
import os
import time
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition

from kinopoisk_classifier.inference.config import InferenceSettings
from kinopoisk_classifier.inference.runtime import ModelRuntime
from kinopoisk_classifier.inference.worker import InferenceWorker
from kinopoisk_classifier.shared.schemas import (
    PredictionEventV1,
    ReviewEventV1,
    make_review_event_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "INFERENCE_KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)
MLFLOW_TRACKING_URI = os.getenv(
    "INFERENCE_MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)
MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "rubert-sentiment")
# Версию требуем указать явно. Так тест не начнёт незаметно проверять другую
# модель после появления новых Registry-версий.
MODEL_VERSION = os.getenv("INFERENCE_MODEL_VERSION")
REVIEW_CSV = Path(
    os.getenv("REAL_REVIEW_CSV", PROJECT_ROOT / "data" / "clean_dataset.csv")
)


def _read_one_real_review(csv_path: Path) -> str:
    """Читает только первую строку датасета, не загружая большой CSV в память."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Real review dataset not found: {csv_path}. "
            "Set REAL_REVIEW_CSV to a CSV file with a text column."
        )

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source), None)

    if row is None or not row.get("text", "").strip():
        raise ValueError(f"Dataset {csv_path} has no non-blank first text row")
    return row["text"]


@unittest.skipUnless(
    os.getenv("RUN_REAL_MODEL_INTEGRATION") == "1" and MODEL_VERSION is not None,
    "set RUN_REAL_MODEL_INTEGRATION=1 and INFERENCE_MODEL_VERSION to run",
)
class KafkaRealModelIntegrationTest(unittest.TestCase):
    def test_real_review_produces_valid_prediction_event(self):
        # Декоратор выше не запускает тест без версии. assert нужен ещё и для
        # читателя/IDE: ниже MODEL_VERSION уже точно является строкой.
        assert MODEL_VERSION is not None

        # Runtime создаём до Kafka-клиентов. Если Registry или MinIO недоступны,
        # тест упадёт на конкретной границе загрузки модели, а не по Kafka timeout.
        runtime = ModelRuntime.from_registry(
            tracking_uri=MLFLOW_TRACKING_URI,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            device="cpu",
            batch_size=1,
        )
        self.assertTrue(
            runtime.metadata.artifact_store_uri.startswith(
                ("mlflow-artifacts:/", "s3://")
            ),
            "The real runtime must load its artifact from MinIO",
        )

        review_id = uuid.uuid4().hex[:24]
        group_suffix = uuid.uuid4().hex
        review = ReviewEventV1(
            schema_version=1,
            event_id=make_review_event_id(review_id),
            review_id=review_id,
            text=_read_one_real_review(REVIEW_CSV),
            emitted_at=datetime.now(timezone.utc),
        )

        settings = InferenceSettings(
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            kafka_group_id=f"real-model-inference-{group_suffix}",
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            model_name=MODEL_NAME,
            model_version=int(MODEL_VERSION),
            device="cpu",
            batch_size=1,
            batch_timeout_ms=100,
            poll_timeout_seconds=2.0,
            delivery_timeout_seconds=15.0,
            # У топика уже может быть история от прошлых запусков теста. Для
            # этого сценария нас интересует только отзыв, который создадим ниже.
            auto_offset_reset="latest",
        )
        worker = InferenceWorker(settings, runtime)

        input_producer = Producer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "enable.idempotence": True,
            }
        )
        output_consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "group.id": f"real-model-prediction-{group_suffix}",
                "auto.offset.reset": "latest",
                "enable.auto.commit": False,
            }
        )

        try:
            worker.consumer.subscribe([settings.input_topic])
            output_consumer.subscribe([settings.output_topic])

            # Одной subscribe() недостаточно: partitions назначаются только во
            # время poll(). Если отправить отзыв раньше, consumer с `latest`
            # может получить стартовую позицию уже ПОСЛЕ него и пропустить тест.
            # Поэтому сначала дожидаемся assignment и фиксируем текущий конец
            # обоих топиков, а уже затем публикуем входное сообщение.
            self._start_from_current_topic_end(
                worker.consumer,
                consumer_name="inference consumer",
            )
            self._start_from_current_topic_end(
                output_consumer,
                consumer_name="prediction consumer",
            )

            input_producer.produce(
                settings.input_topic,
                key=review_id.encode("utf-8"),
                value=review.model_dump_json(exclude_none=True).encode("utf-8"),
            )
            self.assertEqual(input_producer.flush(15.0), 0)

            # Assignment уже готов, но доставка сообщения между producer и
            # broker всё равно асинхронна. Несколько run_once() дают Kafka время
            # вернуть только что опубликованный отзыв.
            inference_deadline = time.monotonic() + 300.0
            while time.monotonic() < inference_deadline:
                if worker.run_once() > 0:
                    break
            else:
                self.fail("InferenceWorker did not consume the real review in time")

            prediction_deadline = time.monotonic() + 60.0
            prediction = self._find_prediction(
                output_consumer,
                review_id=review_id,
                deadline=prediction_deadline,
            )

            # Сам вызов model_validate_json уже доказывает соответствие полному
            # PredictionEventV1: probabilities, confidence, label и id согласованы.
            self.assertEqual(prediction.review_id, review_id)
            self.assertEqual(prediction.source_event_id, review.event_id)
            self.assertEqual(prediction.model.name, MODEL_NAME)
            self.assertEqual(prediction.model.version, MODEL_VERSION)
            self.assertEqual(prediction.model.run_id, runtime.metadata.run_id)
        finally:
            input_producer.flush(5.0)
            output_consumer.close()
            worker.producer.flush(5.0)
            worker.consumer.close()

    def _start_from_current_topic_end(self, consumer, consumer_name):
        """Изолирует тест от сообщений, оставшихся после прошлых запусков.

        Production consumer так не перематываем: там история и committed offset
        обязательны. Здесь каждая consumer group одноразовая, а старые records
        относятся к другим тестовым примерам и только мешают проверке.
        """

        assignment_deadline = time.monotonic() + 30.0
        while time.monotonic() < assignment_deadline:
            message = consumer.poll(0.2)
            if message is not None and message.error():
                raise KafkaException(message.error())

            assignment = consumer.assignment()
            if not assignment:
                continue

            # high watermark — offset следующего сообщения после последнего
            # существующего record. seek(high) означает: старую историю не
            # читаем, но всё опубликованное после этой строки уже увидим.
            for partition in assignment:
                _, high_watermark = consumer.get_watermark_offsets(
                    partition,
                    timeout=5.0,
                )
                consumer.seek(
                    TopicPartition(
                        partition.topic,
                        partition.partition,
                        high_watermark,
                    )
                )
            return

        self.fail(f"Kafka did not assign partitions to {consumer_name} in time")

    def _find_prediction(self, consumer, review_id, deadline):
        """Пропускает старые события и ждёт prediction именно нашего отзыва."""

        while time.monotonic() < deadline:
            message = consumer.poll(1.0)
            if message is None or message.error():
                continue

            prediction = PredictionEventV1.model_validate_json(message.value())
            if prediction.review_id == review_id:
                return prediction

        self.fail("PredictionEventV1 for the real review was not received in time")
