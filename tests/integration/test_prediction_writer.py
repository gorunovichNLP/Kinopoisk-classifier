"""Интеграционный тест PredictionWriter с настоящими Kafka и PostgreSQL.

Тестовые паттерны:
- Component Integration Test: используются реальные broker и database.
- Test Isolation: уникальные prediction_id и Kafka consumer group.
- Arrange–Act–Assert: публикуем событие, запускаем Writer и проверяем эффекты.

GoF-паттерны непосредственно в тестовом файле не используются.
"""

import os
import time
import unittest
import uuid
from datetime import datetime, timezone
from threading import Event, Thread

import psycopg
from confluent_kafka import Producer, TopicPartition

from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings
from kinopoisk_classifier.prediction_writer.postgres import (
    PostgresPredictionRepository,
)
from kinopoisk_classifier.prediction_writer.writer import PredictionWriter
from kinopoisk_classifier.shared.schemas import (
    PredictionEventV1,
    make_prediction_id,
    make_review_event_id,
)


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@unittest.skipUnless(
    os.getenv("RUN_PREDICTION_WRITER_INTEGRATION") == "1",
    "set RUN_PREDICTION_WRITER_INTEGRATION=1 to use Docker Kafka and PostgreSQL",
)
class PredictionWriterIntegrationTest(unittest.TestCase):
    def test_run_saves_prediction_commits_offset_and_stops(self):
        """Runs the writer, commits an event, and stops cleanly."""

        suffix = uuid.uuid4().hex
        review_id = suffix[:24]
        source_event_id = make_review_event_id(review_id)
        model_name = "writer-integration-model"
        model_version = suffix
        prediction_id = make_prediction_id(
            source_event_id,
            model_name,
            model_version,
        )

        prediction = PredictionEventV1(
            schema_version=1,
            prediction_id=prediction_id,
            source_event_id=source_event_id,
            review_id=review_id,
            sentiment="neu",
            label_id=1,
            confidence=0.8,
            probabilities={"neg": 0.1, "neu": 0.8, "pos": 0.1},
            model={
                "name": model_name,
                "version": model_version,
                "run_id": "writer-integration-run",
            },
            predicted_at=datetime.now(timezone.utc),
        )

        settings = PredictionWriterSettings(
            kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            kafka_group_id=f"prediction-writer-integration-{suffix}",
            # Тесту не нужны старые records общего topic. Production default
            # остаётся earliest и при первом запуске обработает всю историю.
            auto_offset_reset="latest",
            poll_timeout_seconds=1.0,
            _env_file=None,
        )
        repository = PostgresPredictionRepository(settings)
        writer = PredictionWriter(settings, repository)

        input_producer = Producer(
            {"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS}
        )
        delivered_messages = []
        delivery_errors = []

        def on_delivery(error, message):
            if error is not None:
                delivery_errors.append(error)
            else:
                delivered_messages.append(message)

        verification_connection = psycopg.connect(
            str(settings.postgres_dsn),
            connect_timeout=settings.postgres_connect_timeout_seconds,
        )

        # run() блокирует поток, поэтому запускаем его в фоне.
        stop_event = Event()
        thread_errors = []

        def run_writer():
            try:
                writer.run(stop_event)
            except BaseException as error:
                # Исключение из Thread не падает в основном тестовом потоке само.
                thread_errors.append(error)

        writer_thread = Thread(target=run_writer, daemon=True)

        try:
            # Сначала poll позволяет Kafka назначить partition новой group.
            # После assignment позиция latest уже зафиксирована; теперь новое
            # тестовое сообщение не будет случайно пропущено.
            assignment_deadline = time.monotonic() + 10.0
            while (
                not writer.consumer.assignment()
                and time.monotonic() < assignment_deadline
            ):
                writer.run_once()
            self.assertTrue(writer.consumer.assignment())

            input_producer.produce(
                settings.input_topic,
                key=review_id.encode("utf-8"),
                value=prediction.model_dump_json(exclude_none=True).encode("utf-8"),
                on_delivery=on_delivery,
            )
            self.assertEqual(input_producer.flush(10.0), 0)
            self.assertEqual(delivery_errors, [])
            self.assertEqual(len(delivered_messages), 1)

            # Запускаем настоящий бесконечный цикл после публикации сообщения.
            writer_thread.start()

            # Ждём видимую закоммиченную строку в отдельном соединении.
            row = None
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if thread_errors:
                    self.fail(f"PredictionWriter failed: {thread_errors[0]!r}")

                with verification_connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT sentiment FROM sentiment.prediction_history "
                        "WHERE prediction_id = %s",
                        (prediction_id,),
                    )
                    row = cursor.fetchone()
                verification_connection.commit()

                if row is not None:
                    break
                time.sleep(0.05)
            else:
                self.fail("PredictionWriter did not save the event in time")

            # Просим цикл остановиться. Если Writer уже начал следующий poll,
            # join подождёт не дольше poll_timeout_seconds плюс небольшой запас.
            stop_event.set()
            writer_thread.join(timeout=5.0)

            self.assertEqual(row, ("neu",))
            self.assertFalse(writer_thread.is_alive())
            self.assertEqual(thread_errors, [])

            # Delivery callback сообщает точные partition и offset сообщения.
            delivered = delivered_messages[0]
            committed = writer.consumer.committed(
                [TopicPartition(settings.input_topic, delivered.partition())],
                timeout=5.0,
            )
            self.assertEqual(committed[0].offset, delivered.offset() + 1)
        finally:
            stop_event.set()
            writer_thread.join(timeout=5.0)
            input_producer.flush(5.0)
            writer.close()
            repository.close()

            with verification_connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM sentiment.prediction_history "
                    "WHERE prediction_id = %s",
                    (prediction_id,),
                )
            verification_connection.commit()
            verification_connection.close()


if __name__ == "__main__":
    unittest.main()
