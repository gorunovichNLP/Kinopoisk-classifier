"""Интеграционный тест repository с настоящим PostgreSQL.

Тестовые паттерны:
- Component Integration Test: repository работает с реальной базой данных.
- Test Isolation: prediction_id уникален, а тест удаляет только свою строку.
- Arrange–Act–Assert: создаём событие, сохраняем и проверяем SQL-результат.

GoF-паттерны непосредственно в тестовом файле не используются.
"""

import os
import unittest
import uuid
from datetime import datetime, timezone

import psycopg

from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings
from kinopoisk_classifier.prediction_writer.postgres import (
    PostgresPredictionRepository,
)
from kinopoisk_classifier.shared.schemas import (
    PredictionEventV1,
    make_prediction_id,
    make_review_event_id,
)


@unittest.skipUnless(
    os.getenv("RUN_POSTGRES_INTEGRATION") == "1",
    "set RUN_POSTGRES_INTEGRATION=1 to use Docker predictions-postgres",
)
class PostgresPredictionRepositoryIntegrationTest(unittest.TestCase):
    def test_saves_prediction_and_ignores_duplicate(self):
        """Saves one prediction and keeps one row after a duplicate."""

        # uuid даёт уникальный review для каждого запуска теста.
        review_id = uuid.uuid4().hex[:24]
        source_event_id = make_review_event_id(review_id)
        model_name = "integration-model"
        model_version = uuid.uuid4().hex

        # prediction_id детерминирован теми же полями, что и в рабочем pipeline.
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
            movie_id="integration-movie",
            source_created_at=datetime.now(timezone.utc),
            sentiment="pos",
            label_id=2,
            confidence=0.8,
            probabilities={"neg": 0.1, "neu": 0.1, "pos": 0.8},
            model={
                "name": model_name,
                "version": model_version,
                "run_id": "integration-run",
            },
            predicted_at=datetime.now(timezone.utc),
        )

        settings = PredictionWriterSettings(_env_file=None)
        repository = PostgresPredictionRepository(settings)

        # Отдельное соединение проверяет только уже закоммиченные данные.
        verification_connection = psycopg.connect(
            str(settings.postgres_dsn),
            connect_timeout=settings.postgres_connect_timeout_seconds,
        )

        try:
            repository.ping()

            first_inserted = repository.save(prediction)
            duplicate_inserted = repository.save(prediction)

            with verification_connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT sentiment, model_name, model_version, COUNT(*)
                    FROM sentiment.prediction_history
                    WHERE prediction_id = %s
                    GROUP BY sentiment, model_name, model_version
                    """,
                    (prediction_id,),
                )
                row = cursor.fetchone()

            verification_connection.commit()

            self.assertTrue(first_inserted)
            self.assertFalse(duplicate_inserted)
            self.assertEqual(
                row,
                ("pos", model_name, model_version, 1),
            )
        finally:
            # Удаляем только строку этого теста и не трогаем историю приложения.
            with verification_connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM sentiment.prediction_history "
                    "WHERE prediction_id = %s",
                    (prediction_id,),
                )
            verification_connection.commit()
            verification_connection.close()
            repository.close()


if __name__ == "__main__":
    unittest.main()
