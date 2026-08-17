"""Environment-based configuration for the review producer."""

from pydantic import MongoDsn, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from kinopoisk_classifier.shared.schemas import NonBlankString


class ReviewProducerSettings(BaseSettings):
    """Validated configuration for the review producer."""

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_PRODUCER_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )


    mongo_uri: MongoDsn = (
        "mongodb://kinopoisk:kinopoisk@localhost:27017/kinopoisk?authSource=admin"
    )
    mongo_database: NonBlankString = "kinopoisk"
    reviews_collection: NonBlankString = "reviews"
    checkpoints_collection: NonBlankString = "pipeline_checkpoints"



    checkpoint_id: NonBlankString = "reviews-to-kafka-v1"

    kafka_bootstrap_servers: NonBlankString = "localhost:9092"
    kafka_client_id: NonBlankString = "kinopoisk-review-producer"
    output_topic: NonBlankString = "kinopoisk.reviews.v1"

    batch_size: PositiveInt = 50
    poll_interval_seconds: PositiveFloat = 1.0
    delivery_timeout_seconds: PositiveFloat = 30.0
    mongo_server_selection_timeout_ms: PositiveInt = 5_000

    def producer_config(self) -> dict:
        """Build an idempotent Kafka producer configuration."""

        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "client.id": self.kafka_client_id,
            "enable.idempotence": True,
            "acks": "all",
        }
