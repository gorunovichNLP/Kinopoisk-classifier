"""Integration tests for the Kafka inference worker."""

import os
import time
import unittest
import uuid
from datetime import datetime, timezone

from confluent_kafka import Consumer, Producer

from kinopoisk_classifier.inference.config import InferenceSettings
from kinopoisk_classifier.inference.worker import InferenceWorker
from kinopoisk_classifier.shared.schemas import (
    LoadedModelMetadata,
    PredictionEventV1,
    ReviewEventV1,
    SentimentPrediction,
    make_review_event_id,
)


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


class _FakeRuntime:
    """_FakeRuntime implementation."""

    metadata = LoadedModelMetadata(
        name="rubert-sentiment",
        version="integration-test",
        run_id="integration-test-run",
        registry_uri="models:/rubert-sentiment/integration-test",
        artifact_store_uri="mlflow-artifacts:/integration-test/artifacts",
    )

    def predict_batch(self, texts):
        return [
            SentimentPrediction(
                sentiment="pos",
                label_id=2,
                confidence=0.8,
                probabilities={"neg": 0.1, "neu": 0.1, "pos": 0.8},
            )
            for _ in texts
        ]


@unittest.skipUnless(
    os.getenv("RUN_KAFKA_INTEGRATION") == "1",
    "set RUN_KAFKA_INTEGRATION=1 to use the Docker Kafka broker",
)
class KafkaWorkerIntegrationTest(unittest.TestCase):
    def test_review_is_consumed_and_prediction_is_published(self):

        review_id = uuid.uuid4().hex[:24]
        group_suffix = uuid.uuid4().hex
        review = ReviewEventV1(
            schema_version=1,
            event_id=make_review_event_id(review_id),
            review_id=review_id,
            text="Integration review",
            emitted_at=datetime.now(timezone.utc),
        )

        settings = InferenceSettings(
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            kafka_group_id=f"inference-integration-{group_suffix}",
            model_version=1,
            batch_size=8,
            batch_timeout_ms=100,
            poll_timeout_seconds=2.0,
            delivery_timeout_seconds=10.0,
        )
        worker = InferenceWorker(settings, _FakeRuntime())

        input_producer = Producer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "enable.idempotence": True,
            }
        )
        output_consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
                "group.id": f"prediction-integration-{group_suffix}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )

        try:


            worker.consumer.subscribe([settings.input_topic])
            output_consumer.subscribe([settings.output_topic])

            input_producer.produce(
                settings.input_topic,
                key=review_id.encode("utf-8"),
                value=review.model_dump_json(exclude_none=True).encode("utf-8"),
            )
            self.assertEqual(input_producer.flush(10.0), 0)



            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if worker.run_once() > 0:
                    break
            else:
                self.fail("InferenceWorker did not consume the review in time")

            prediction = self._find_prediction(
                output_consumer,
                review_id=review_id,
                deadline=deadline,
            )
            self.assertEqual(prediction.sentiment, "pos")
            self.assertEqual(prediction.source_event_id, review.event_id)
            self.assertEqual(prediction.model.version, "integration-test")
        finally:
            input_producer.flush(5.0)
            output_consumer.close()
            worker.producer.flush(5.0)
            worker.consumer.close()

    def _find_prediction(self, consumer, review_id, deadline):
        while time.monotonic() < deadline:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                continue

            event = PredictionEventV1.model_validate_json(message.value())
            if event.review_id == review_id:
                return event

        self.fail("PredictionEventV1 was not received in time")
