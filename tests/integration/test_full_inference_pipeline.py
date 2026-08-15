r"""Сквозной интеграционный тест всего inference-контура.

Проверяем один настоящий отзыв по полному маршруту:

    MongoDB -> ReviewProducer -> Kafka reviews -> InferenceWorker
            -> модель из MLflow/MinIO -> Kafka predictions
            -> PredictionWriter -> PostgreSQL

Тестовые паттерны:
- End-to-End Integration Test: вместе работают все настоящие компоненты контура.
- Arrange-Act-Assert: подготавливаем отзыв, проводим его по этапам и проверяем
  checkpoint вместе с итоговой строкой PostgreSQL.
- Test Isolation: каждый запуск получает уникальные MongoDB-коллекции и Kafka
  consumer groups, поэтому параллельные и предыдущие запуски не смешиваются.

GoF-паттерны непосредственно в тестовом файле не реализуются.

Тест тяжёлый и выключен по умолчанию. Перед запуском нужно поднять Docker
Compose и указать существующую версию модели в MLflow Registry:

    docker compose -f infra/docker/docker-compose.yml up -d
    $env:RUN_REAL_MODEL_INTEGRATION="1"
    $env:INFERENCE_MODEL_VERSION="2"
    venv\Scripts\python -m unittest tests.integration.test_full_inference_pipeline -v
"""

# csv нужен, чтобы взять один настоящий текст из обучающего датасета.
import csv
# os читает флаги запуска и адреса локальной инфраструктуры.
import os
# time используется для понятных ограниченных циклов ожидания Kafka.
import time
# unittest предоставляет проверки и возможность пропустить тяжёлый тест.
import unittest
# uuid создаёт уникальные имена коллекций и consumer groups для каждого запуска.
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ObjectId — тот же тип идентификатора, который используют реальные отзывы.
from bson import ObjectId
# KafkaException сообщает о настоящей ошибке broker-а во время poll().
# TopicPartition нужен, чтобы начать тест строго с текущего конца topic.
from confluent_kafka import KafkaException, TopicPartition
# MongoClient подготавливает исходный отзыв и удаляет тестовые коллекции.
from pymongo import MongoClient

from kinopoisk_classifier.inference.config import InferenceSettings
from kinopoisk_classifier.inference.runtime import ModelRuntime
from kinopoisk_classifier.inference.worker import InferenceWorker
from kinopoisk_classifier.prediction_writer.config import (
    PredictionWriterSettings,
)
from kinopoisk_classifier.prediction_writer.postgres import (
    PostgresPredictionRepository,
)
from kinopoisk_classifier.prediction_writer.writer import PredictionWriter
from kinopoisk_classifier.review_producer.checkpoint import MongoCheckpointStore
from kinopoisk_classifier.review_producer.config import ReviewProducerSettings
from kinopoisk_classifier.review_producer.kafka import KafkaReviewPublisher
from kinopoisk_classifier.review_producer.mongo import MongoReviewReader
from kinopoisk_classifier.review_producer.producer import ReviewProducer
from kinopoisk_classifier.shared.schemas import (
    make_prediction_id,
    make_review_event_id,
)


# Корень проекта нужен для стандартного пути к датасету.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Все сервисы, запущенные с Windows, обращаются к Kafka через host listener.
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "INFERENCE_KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

# MLflow хранит Registry-метаданные, а артефакты модели отдаёт из MinIO.
MLFLOW_TRACKING_URI = os.getenv(
    "INFERENCE_MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)
MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "rubert-sentiment")

# Версию нельзя выбирать неявно: тест должен быть воспроизводимым.
MODEL_VERSION = os.getenv("INFERENCE_MODEL_VERSION")

# При необходимости путь можно заменить переменной REAL_REVIEW_CSV.
REVIEW_CSV = Path(
    os.getenv("REAL_REVIEW_CSV", PROJECT_ROOT / "data" / "clean_dataset.csv")
)


def _read_one_real_review(csv_path: Path) -> str:
    """Читает первый непустой текст, не загружая весь CSV в память."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Real review dataset not found: {csv_path}. "
            "Set REAL_REVIEW_CSV to a CSV file with a text column."
        )

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            text = row.get("text", "").strip()
            if text:
                return text

    raise ValueError(f"Dataset {csv_path} has no non-blank text rows")


@unittest.skipUnless(
    os.getenv("RUN_REAL_MODEL_INTEGRATION") == "1" and MODEL_VERSION is not None,
    "set RUN_REAL_MODEL_INTEGRATION=1 and INFERENCE_MODEL_VERSION to run",
)
class FullInferencePipelineIntegrationTest(unittest.TestCase):
    def test_review_travels_from_mongodb_to_postgresql(self):
        """Проводит один отзыв через все настоящие границы приложения."""

        # Декоратор не запустит тест без версии. assert дополнительно объясняет
        # Python и IDE, что ниже MODEL_VERSION уже точно является строкой.
        assert MODEL_VERSION is not None

        # Один suffix изолирует все временные имена этого запуска.
        suffix = uuid.uuid4().hex
        reviews_collection = f"reviews_full_pipeline_{suffix}"
        checkpoints_collection = f"checkpoints_full_pipeline_{suffix}"
        checkpoint_id = f"full-pipeline-{suffix}"

        # ObjectId заранее создаём в тесте, чтобы затем проследить один и тот же
        # review_id через MongoDB, Kafka и PostgreSQL.
        review_object_id = ObjectId()
        review_id = str(review_object_id)
        movie_id = f"integration-movie-{suffix}"
        source_created_at = datetime.now(timezone.utc)

        # _env_file=None защищает тест от случайных значений из .env.local.
        review_settings = ReviewProducerSettings(
            reviews_collection=reviews_collection,
            checkpoints_collection=checkpoints_collection,
            checkpoint_id=checkpoint_id,
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            kafka_client_id=f"full-pipeline-review-producer-{suffix}",
            batch_size=1,
            delivery_timeout_seconds=30.0,
            _env_file=None,
        )
        inference_settings = InferenceSettings(
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            kafka_client_id=f"full-pipeline-inference-{suffix}",
            kafka_group_id=f"full-pipeline-inference-{suffix}",
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            model_name=MODEL_NAME,
            model_version=int(MODEL_VERSION),
            device="cpu",
            batch_size=1,
            batch_timeout_ms=100,
            poll_timeout_seconds=2.0,
            delivery_timeout_seconds=30.0,
            auto_offset_reset="latest",
            _env_file=None,
        )
        writer_settings = PredictionWriterSettings(
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            kafka_client_id=f"full-pipeline-writer-{suffix}",
            kafka_group_id=f"full-pipeline-writer-{suffix}",
            poll_timeout_seconds=2.0,
            auto_offset_reset="latest",
            _env_file=None,
        )

        # Этот общий MongoClient принадлежит тесту. Reader и checkpoint store
        # получают его через Dependency Injection и сами его не закрывают.
        mongo_client = MongoClient(
            str(review_settings.mongo_uri),
            tz_aware=True,
            serverSelectionTimeoutMS=(
                review_settings.mongo_server_selection_timeout_ms
            ),
        )
        database = mongo_client[review_settings.mongo_database]

        # Переменные объявлены до try, чтобы finally был безопасен даже при
        # ошибке во время создания одного из следующих компонентов.
        publisher = None
        inference_worker = None
        repository = None
        prediction_writer = None

        try:
            # Сначала проверяем дешёвые подключения к базам. Если Docker не
            # поднят, тест не будет зря несколько минут загружать модель.
            mongo_client.admin.command("ping")
            repository = PostgresPredictionRepository(writer_settings)
            repository.ping()

            # Настоящая модель загружается через MLflow Registry. Сам MLflow
            # получает её файлы из настроенного MinIO artifact store.
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

            # Собираем три настоящих application service. В отличие от main(),
            # тест вызывает run_once(), чтобы явно видеть каждый этап pipeline.
            reader = MongoReviewReader(review_settings, client=mongo_client)
            checkpoint_store = MongoCheckpointStore(
                review_settings,
                client=mongo_client,
            )
            publisher = KafkaReviewPublisher(review_settings)
            review_producer = ReviewProducer(
                review_settings,
                reader,
                publisher,
                checkpoint_store,
            )

            inference_worker = InferenceWorker(inference_settings, runtime)
            inference_worker.consumer.subscribe([inference_settings.input_topic])

            prediction_writer = PredictionWriter(writer_settings, repository)

            # subscribe() только объявляет интерес к topic. Реальное назначение
            # partitions происходит во время poll(), поэтому сначала ждём его.
            # Одновременно ставим оба consumer-а на текущий конец topic: старые
            # тестовые события нам не нужны, а новые после этой строки не потеряются.
            self._start_from_current_topic_end(
                inference_worker.consumer,
                consumer_name="inference consumer",
            )
            self._start_from_current_topic_end(
                prediction_writer.consumer,
                consumer_name="prediction writer consumer",
            )

            # Только после готовности обоих Kafka consumer-ов создаём источник.
            # Отзыв неизменяем: тест больше не обновляет этот документ.
            database[reviews_collection].insert_one(
                {
                    "_id": review_object_id,
                    "movie_id": movie_id,
                    "text": _read_one_real_review(REVIEW_CSV),
                    "created_at": source_created_at,
                }
            )

            # Этап 1: MongoDB -> Kafka reviews -> MongoDB checkpoint.
            self.assertEqual(review_producer.run_once(), 1)
            checkpoint = checkpoint_store.load()
            self.assertIsNotNone(checkpoint)
            self.assertEqual(checkpoint.last_review_id, review_id)

            # Этап 2: Kafka reviews -> настоящая модель -> Kafka predictions.
            # На слабом компьютере один predict может выполняться долго, поэтому
            # на получение входного события даём пять минут, как в отдельном
            # real-model integration test.
            inference_deadline = time.monotonic() + 300.0
            while time.monotonic() < inference_deadline:
                if inference_worker.run_once() > 0:
                    break
            else:
                self.fail("InferenceWorker did not process the review in time")

            # Этап 3: Kafka predictions -> PostgreSQL. Здесь модель уже считать
            # не должна, поэтому одной минуты достаточно для доставки Kafka.
            writer_deadline = time.monotonic() + 60.0
            while time.monotonic() < writer_deadline:
                if prediction_writer.run_once() > 0:
                    break
            else:
                self.fail("PredictionWriter did not save the prediction in time")

            # InferenceWorker строит один и тот же prediction_id из события и
            # версии модели. По этому стабильному ключу проверяем итоговую строку.
            expected_prediction_id = make_prediction_id(
                make_review_event_id(review_id),
                runtime.metadata.name,
                runtime.metadata.version,
            )

            with repository.connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        source_event_id,
                        review_id,
                        movie_id,
                        sentiment,
                        label_id,
                        confidence,
                        model_name,
                        model_version,
                        model_run_id
                    FROM sentiment.prediction_history
                    WHERE prediction_id = %s
                    """,
                    (expected_prediction_id,),
                )
                saved_prediction = cursor.fetchone()
            repository.connection.commit()

            # PostgreSQL constraints уже проверили вероятности и соответствие
            # label/sentiment. Здесь проверяем сквозное сохранение идентичности.
            self.assertIsNotNone(saved_prediction)
            self.assertEqual(saved_prediction[0], make_review_event_id(review_id))
            self.assertEqual(saved_prediction[1], review_id)
            self.assertEqual(saved_prediction[2], movie_id)
            self.assertIn(saved_prediction[3], ("neg", "neu", "pos"))
            self.assertIn(saved_prediction[4], (0, 1, 2))
            self.assertGreaterEqual(saved_prediction[5], 0.0)
            self.assertLessEqual(saved_prediction[5], 1.0)
            self.assertEqual(saved_prediction[6], runtime.metadata.name)
            self.assertEqual(saved_prediction[7], runtime.metadata.version)
            self.assertEqual(saved_prediction[8], runtime.metadata.run_id)
        finally:
            # Удаляем только строку этого уникального отзыва. История других
            # ручных запусков и тестов остаётся нетронутой.
            if repository is not None:
                try:
                    with repository.connection.cursor() as cursor:
                        cursor.execute(
                            "DELETE FROM sentiment.prediction_history "
                            "WHERE review_id = %s",
                            (review_id,),
                        )
                    repository.connection.commit()
                except Exception:
                    repository.connection.rollback()

            # Закрываем ресурсы в обратном порядке их использования.
            if prediction_writer is not None:
                prediction_writer.close()
            if repository is not None:
                repository.close()
            if inference_worker is not None:
                inference_worker.producer.flush(5.0)
                inference_worker.consumer.close()
            if publisher is not None:
                publisher.producer.flush(5.0)

            # Коллекции имеют уникальные имена и созданы только этим тестом.
            database.drop_collection(reviews_collection)
            database.drop_collection(checkpoints_collection)
            mongo_client.close()

    def _start_from_current_topic_end(self, consumer, consumer_name):
        """Дожидается partitions и пропускает историю прошлых запусков."""

        assignment_deadline = time.monotonic() + 30.0
        while time.monotonic() < assignment_deadline:
            # poll запускает присоединение consumer-а к группе и assignment.
            message = consumer.poll(0.2)
            if message is not None and message.error():
                raise KafkaException(message.error())

            assignment = consumer.assignment()
            if not assignment:
                continue

            # high watermark — offset после последнего существующего сообщения.
            # seek(high) оставляет старую историю позади, но все новые records,
            # опубликованные после этого места, consumer уже увидит.
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
