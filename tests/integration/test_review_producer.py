"""Интеграционный тест полного ReviewProducer.run().

Тестовые паттерны:
- Component Integration Test: для границ ReviewProducer используются настоящие
  MongoDB и Kafka; inference worker и модель в этот тест не входят.
- Arrange–Act–Assert: готовим отзыв, запускаем цикл и проверяем событие/checkpoint.
- Test Isolation: каждый запуск использует уникальные MongoDB-коллекции и id.

GoF-паттерны непосредственно в тестовом файле не реализуются.
"""

import os
import time
import unittest
import uuid
from datetime import datetime, timezone
from threading import Event, Thread

from bson import ObjectId
from confluent_kafka import Consumer
from pymongo import MongoClient

from kinopoisk_classifier.review_producer.checkpoint import MongoCheckpointStore
from kinopoisk_classifier.review_producer.config import ReviewProducerSettings
from kinopoisk_classifier.review_producer.kafka import KafkaReviewPublisher
from kinopoisk_classifier.review_producer.mongo import MongoReviewReader
from kinopoisk_classifier.review_producer.producer import ReviewProducer
from kinopoisk_classifier.shared.schemas import ReviewEventV1


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@unittest.skipUnless(
    os.getenv("RUN_REVIEW_PRODUCER_INTEGRATION") == "1",
    "set RUN_REVIEW_PRODUCER_INTEGRATION=1 to use Docker MongoDB and Kafka",
)
class ReviewProducerIntegrationTest(unittest.TestCase):
    def test_run_publishes_review_saves_checkpoint_and_stops(self):
        # Уникальные имена изолируют MongoDB-данные этого запуска теста.
        suffix = uuid.uuid4().hex
        reviews_collection = f"reviews_producer_integration_{suffix}"
        checkpoints_collection = f"checkpoints_producer_integration_{suffix}"
        review_id = ObjectId()

        settings = ReviewProducerSettings(
            reviews_collection=reviews_collection,
            checkpoints_collection=checkpoints_collection,
            checkpoint_id=f"review-producer-integration-{suffix}",
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            batch_size=5,
            poll_interval_seconds=0.1,
            delivery_timeout_seconds=10.0,
            _env_file=None,
        )

        # Отдельный client наполняет MongoDB и позже удаляет тестовые коллекции.
        seed_client = MongoClient(str(settings.mongo_uri), tz_aware=True)
        seed_client[settings.mongo_database][reviews_collection].insert_one(
            {
                "_id": review_id,
                "text": "Интеграционный отзыв полного ReviewProducer",
                "created_at": datetime.now(timezone.utc),
            }
        )

        reader = MongoReviewReader(settings)
        publisher = KafkaReviewPublisher(settings)
        checkpoint_store = MongoCheckpointStore(settings)
        review_producer = ReviewProducer(
            settings,
            reader,
            publisher,
            checkpoint_store,
        )

        # Уникальная consumer group читает событие, созданное этим тестом.
        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "group.id": f"review-producer-integration-{suffix}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([settings.output_topic])

        stop_event = Event()
        thread_errors = []

        # run() блокирует текущий поток, поэтому тест запускает его в фоне.
        def run_producer():
            try:
                review_producer.run(stop_event)
            except BaseException as error:
                # Исключение внутри Thread само не падает в основном потоке теста.
                # Сохраняем его и проверяем после join().
                thread_errors.append(error)

        producer_thread = Thread(target=run_producer, daemon=True)

        try:
            producer_thread.start()

            received_event = None
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if thread_errors:
                    self.fail(f"ReviewProducer failed: {thread_errors[0]!r}")

                message = consumer.poll(1.0)
                if message is None or message.error():
                    continue
                if message.key() != str(review_id).encode("utf-8"):
                    continue

                received_event = ReviewEventV1.model_validate_json(message.value())
                break

            self.assertIsNotNone(received_event, "ReviewEventV1 was not received")
            self.assertEqual(received_event.review_id, str(review_id))

            # set() немедленно разбудит run(), даже если он ждёт пустой batch.
            stop_event.set()
            producer_thread.join(timeout=5.0)

            self.assertFalse(producer_thread.is_alive())
            self.assertEqual(thread_errors, [])

            checkpoint = checkpoint_store.load()
            self.assertIsNotNone(checkpoint)
            self.assertEqual(checkpoint.last_review_id, str(review_id))
        finally:
            stop_event.set()
            producer_thread.join(timeout=5.0)
            consumer.close()
            publisher.producer.flush(5.0)
            reader.close()
            checkpoint_store.close()
            database = seed_client[settings.mongo_database]
            database.drop_collection(reviews_collection)
            database.drop_collection(checkpoints_collection)
            seed_client.close()


if __name__ == "__main__":
    unittest.main()
