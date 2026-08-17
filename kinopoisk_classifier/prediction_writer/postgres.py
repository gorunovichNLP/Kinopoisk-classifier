"""Persist prediction events in PostgreSQL."""


import psycopg


from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings


from kinopoisk_classifier.shared.schemas import PredictionEventV1





INSERT_PREDICTION_SQL = """
INSERT INTO sentiment.prediction_history (
    prediction_id,
    schema_version,
    source_event_id,
    review_id,
    movie_id,
    source_created_at,
    sentiment,
    label_id,
    confidence,
    probability_neg,
    probability_neu,
    probability_pos,
    model_name,
    model_version,
    model_run_id,
    predicted_at
)
VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s
)
ON CONFLICT (prediction_id) DO NOTHING
"""


class PostgresPredictionRepository:
    """Repository for idempotent prediction persistence."""

    def __init__(
        self,
        settings: PredictionWriterSettings,
        *,
        connection=None,
    ) -> None:
        self.settings = settings



        self._owns_connection = connection is None

        if connection is None:

            self.connection = psycopg.connect(
                str(settings.postgres_dsn),
                connect_timeout=settings.postgres_connect_timeout_seconds,
            )
        else:
            self.connection = connection

    def ping(self) -> None:
        """Verify the backing service connection."""

        try:

            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")



            self.connection.commit()
        except Exception:

            self.connection.rollback()
            raise

    def save(self, prediction: PredictionEventV1) -> bool:
        """Persist a validated value transactionally."""


        values = (
            prediction.prediction_id,
            prediction.schema_version,
            prediction.source_event_id,
            prediction.review_id,
            prediction.movie_id,
            prediction.source_created_at,
            prediction.sentiment,
            prediction.label_id,
            prediction.confidence,
            prediction.probabilities.neg,
            prediction.probabilities.neu,
            prediction.probabilities.pos,
            prediction.model.name,
            prediction.model.version,
            prediction.model.run_id,
            prediction.predicted_at,
        )

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(INSERT_PREDICTION_SQL, values)



                inserted = cursor.rowcount == 1


            self.connection.commit()
            return inserted
        except Exception:

            self.connection.rollback()
            raise

    def close(self) -> None:
        """Close resources owned by this instance."""

        if self._owns_connection:
            self.connection.close()
