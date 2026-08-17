"""Integration tests for the review producer service."""

import os
import time
import unittest
import uuid
from datetime import datetime, timezone
from threading import Event, Thread
from unittest.mock import patch

from bson import ObjectId
from confluent_kafka import Consumer
from pymongo import MongoClient

from kinopoisk_classifier.review_producer.__main__ import main
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
    def test_main_builds_application_publishes_review_and_stops(self):
        """Exercises the real Composition Root, settings, and cleanup."""


        suffix = uuid.uuid4().hex
        reviews_collection = f"reviews_main_smoke_{suffix}"
        checkpoints_collection = f"checkpoints_main_smoke_{suffix}"
        checkpoint_id = f"review-producer-main-smoke-{suffix}"
        review_id = ObjectId()



        base_settings = ReviewProducerSettings(_env_file=None)



        environment = {
            "REVIEW_PRODUCER_MONGO_URI": str(base_settings.mongo_uri),
            "REVIEW_PRODUCER_MONGO_DATABASE": base_settings.mongo_database,
            "REVIEW_PRODUCER_REVIEWS_COLLECTION": reviews_collection,
            "REVIEW_PRODUCER_CHECKPOINTS_COLLECTION": checkpoints_collection,
            "REVIEW_PRODUCER_CHECKPOINT_ID": checkpoint_id,
            "REVIEW_PRODUCER_KAFKA_BOOTSTRAP_SERVERS": KAFKA_BOOTSTRAP_SERVERS,
            "REVIEW_PRODUCER_KAFKA_CLIENT_ID": f"main-smoke-{suffix}",
            "REVIEW_PRODUCER_OUTPUT_TOPIC": base_settings.output_topic,
            "REVIEW_PRODUCER_BATCH_SIZE": "5",
            "REVIEW_PRODUCER_POLL_INTERVAL_SECONDS": "0.1",
            "REVIEW_PRODUCER_DELIVERY_TIMEOUT_SECONDS": "10",
        }


        seed_client = MongoClient(str(base_settings.mongo_uri), tz_aware=True)
        database = seed_client[base_settings.mongo_database]


        database[reviews_collection].insert_one(
            {
                "_id": review_id,
                "text": "Smoke test for the real Review Producer entry point",
                "created_at": datetime.now(timezone.utc),
            }
        )


        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "group.id": f"review-producer-main-smoke-{suffix}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([base_settings.output_topic])


        stop_event = Event()


        thread_errors = []

        def run_main():
            try:

                main(stop_event)
            except BaseException as error:
                thread_errors.append(error)

        main_thread = Thread(target=run_main, daemon=True)

        try:


            with patch.dict(os.environ, environment, clear=False):
                main_thread.start()

                received_event = None
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:

                    if thread_errors:
                        self.fail(f"ReviewProducer main failed: {thread_errors[0]!r}")

                    message = consumer.poll(1.0)
                    if message is None or message.error():
                        continue



                    if message.key() != str(review_id).encode("utf-8"):
                        continue

                    received_event = ReviewEventV1.model_validate_json(
                        message.value()
                    )
                    break

                self.assertIsNotNone(
                    received_event,
                    "ReviewEventV1 from main() was not received",
                )
                self.assertEqual(received_event.review_id, str(review_id))


                stop_event.set()
                main_thread.join(timeout=5.0)


            self.assertFalse(main_thread.is_alive())
            self.assertEqual(thread_errors, [])


            raw_checkpoint = database[checkpoints_collection].find_one(
                {"_id": checkpoint_id}
            )
            self.assertIsNotNone(raw_checkpoint)
            self.assertEqual(raw_checkpoint["last_review_id"], review_id)

        finally:

            stop_event.set()
            main_thread.join(timeout=5.0)
            consumer.close()
            database.drop_collection(reviews_collection)
            database.drop_collection(checkpoints_collection)
            seed_client.close()

    def test_run_publishes_review_saves_checkpoint_and_stops(self):

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


        seed_client = MongoClient(str(settings.mongo_uri), tz_aware=True)
        seed_client[settings.mongo_database][reviews_collection].insert_one(
            {
                "_id": review_id,
                "text": "Integration review for the complete ReviewProducer",
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


        def run_producer():
            try:
                review_producer.run(stop_event)
            except BaseException as error:


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
