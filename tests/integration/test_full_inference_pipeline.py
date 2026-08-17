"""End-to-end integration test for the complete inference pipeline."""


import csv

import os

import time

import unittest

import uuid
from datetime import datetime, timezone
from pathlib import Path


from bson import ObjectId


from confluent_kafka import KafkaException, TopicPartition

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


MODEL_VERSION = os.getenv("INFERENCE_MODEL_VERSION")


REVIEW_CSV = Path(
    os.getenv("REAL_REVIEW_CSV", PROJECT_ROOT / "data" / "clean_dataset.csv")
)


def _read_one_real_review(csv_path: Path) -> str:
    """Read one non-blank review from a CSV dataset."""

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
        """Perform the test review travels from mongodb to postgresql operation."""



        assert MODEL_VERSION is not None


        suffix = uuid.uuid4().hex
        reviews_collection = f"reviews_full_pipeline_{suffix}"
        checkpoints_collection = f"checkpoints_full_pipeline_{suffix}"
        checkpoint_id = f"full-pipeline-{suffix}"



        review_object_id = ObjectId()
        review_id = str(review_object_id)
        movie_id = f"integration-movie-{suffix}"
        source_created_at = datetime.now(timezone.utc)


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



        mongo_client = MongoClient(
            str(review_settings.mongo_uri),
            tz_aware=True,
            serverSelectionTimeoutMS=(
                review_settings.mongo_server_selection_timeout_ms
            ),
        )
        database = mongo_client[review_settings.mongo_database]



        publisher = None
        inference_worker = None
        repository = None
        prediction_writer = None

        try:


            mongo_client.admin.command("ping")
            repository = PostgresPredictionRepository(writer_settings)
            repository.ping()



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





            self._start_from_current_topic_end(
                inference_worker.consumer,
                consumer_name="inference consumer",
            )
            self._start_from_current_topic_end(
                prediction_writer.consumer,
                consumer_name="prediction writer consumer",
            )



            database[reviews_collection].insert_one(
                {
                    "_id": review_object_id,
                    "movie_id": movie_id,
                    "text": _read_one_real_review(REVIEW_CSV),
                    "created_at": source_created_at,
                }
            )


            self.assertEqual(review_producer.run_once(), 1)
            checkpoint = checkpoint_store.load()
            self.assertIsNotNone(checkpoint)
            self.assertEqual(checkpoint.last_review_id, review_id)




            # real-model integration test.
            inference_deadline = time.monotonic() + 300.0
            while time.monotonic() < inference_deadline:
                if inference_worker.run_once() > 0:
                    break
            else:
                self.fail("InferenceWorker did not process the review in time")



            writer_deadline = time.monotonic() + 60.0
            while time.monotonic() < writer_deadline:
                if prediction_writer.run_once() > 0:
                    break
            else:
                self.fail("PredictionWriter did not save the prediction in time")



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


            if prediction_writer is not None:
                prediction_writer.close()
            if repository is not None:
                repository.close()
            if inference_worker is not None:
                inference_worker.producer.flush(5.0)
                inference_worker.consumer.close()
            if publisher is not None:
                publisher.producer.flush(5.0)


            database.drop_collection(reviews_collection)
            database.drop_collection(checkpoints_collection)
            mongo_client.close()

    def _start_from_current_topic_end(self, consumer, consumer_name):
        """Position a test consumer at the current topic end."""

        assignment_deadline = time.monotonic() + 30.0
        while time.monotonic() < assignment_deadline:

            message = consumer.poll(0.2)
            if message is not None and message.error():
                raise KafkaException(message.error())

            assignment = consumer.assignment()
            if not assignment:
                continue




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
