"""Интеграционный тест Review Publisher с настоящим Kafka broker."""

import os
import time
import unittest
import uuid
from datetime import datetime, timezone

from confluent_kafka import Consumer

from kinopoisk_classifier.review_producer.config import ReviewProducerSettings
from kinopoisk_classifier.review_producer.kafka import KafkaReviewPublisher
from kinopoisk_classifier.shared.schemas import MongoReview, ReviewEventV1


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@unittest.skipUnless(
    os.getenv("RUN_KAFKA_INTEGRATION") == "1",
    "set RUN_KAFKA_INTEGRATION=1 to use the Docker Kafka broker",
)
class KafkaReviewPublisherIntegrationTest(unittest.TestCase):
    def test_publishes_valid_review_event(self):
        review_id = uuid.uuid4().hex[:24]
        settings = ReviewProducerSettings(
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            delivery_timeout_seconds=10.0,
            _env_file=None,
        )
        publisher = KafkaReviewPublisher(settings)
        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "group.id": f"review-publisher-integration-{uuid.uuid4().hex}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([settings.output_topic])

        review = MongoReview.model_validate(
            {
                "_id": review_id,
                "text": "Отзыв для проверки MongoDB → Kafka",
                "created_at": datetime.now(timezone.utc),
            }
        )

        try:
            publisher.publish_batch([review])

            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                message = consumer.poll(1.0)
                if message is None or message.error():
                    continue
                if message.key() != review_id.encode("utf-8"):
                    continue

                event = ReviewEventV1.model_validate_json(message.value())
                self.assertEqual(event.review_id, review_id)
                self.assertEqual(event.text, review.text)
                break
            else:
                self.fail("ReviewEventV1 was not received in time")
        finally:
            consumer.close()
            publisher.producer.flush(5.0)


if __name__ == "__main__":
    unittest.main()
