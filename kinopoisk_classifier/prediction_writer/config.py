"""Environment-based configuration for the prediction writer."""


from typing import Literal




from pydantic import PositiveFloat, PositiveInt, PostgresDsn



from pydantic_settings import BaseSettings, SettingsConfigDict


from kinopoisk_classifier.shared.schemas import NonBlankString


class PredictionWriterSettings(BaseSettings):
    """Validated configuration for the prediction writer."""



    model_config = SettingsConfigDict(
        env_prefix="PREDICTION_WRITER_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )


    kafka_bootstrap_servers: NonBlankString = "localhost:9092"
    kafka_client_id: NonBlankString = "kinopoisk-prediction-writer"
    kafka_group_id: NonBlankString = "kinopoisk-prediction-writer-v1"
    input_topic: NonBlankString = "kinopoisk.predictions.v1"


    auto_offset_reset: Literal["earliest", "latest", "error"] = "earliest"


    poll_timeout_seconds: PositiveFloat = 1.0


    postgres_dsn: PostgresDsn = (
        "postgresql://kinopoisk:kinopoisk@localhost:5433/kinopoisk_predictions"
    )


    postgres_connect_timeout_seconds: PositiveInt = 5

    def consumer_config(self) -> dict:
        """Build a Kafka consumer configuration."""

        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "client.id": self.kafka_client_id,
            "group.id": self.kafka_group_id,
            "auto.offset.reset": self.auto_offset_reset,

            "enable.auto.commit": False,
            "enable.auto.offset.store": False,

            "isolation.level": "read_committed",
        }
