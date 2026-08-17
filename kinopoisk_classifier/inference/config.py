"""Environment-based configuration for the Kafka inference worker."""

from typing import Literal

from pydantic import AnyHttpUrl, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from kinopoisk_classifier.shared.schemas import NonBlankString


class InferenceSettings(BaseSettings):
    """InferenceSettings implementation."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kafka_bootstrap_servers: NonBlankString = "localhost:9092"
    kafka_client_id: NonBlankString = "kinopoisk-sentiment-inference"
    kafka_group_id: NonBlankString = "kinopoisk-sentiment-inference-v1"
    input_topic: NonBlankString = "kinopoisk.reviews.v1"
    output_topic: NonBlankString = "kinopoisk.predictions.v1"
    dlq_topic: NonBlankString = "kinopoisk.reviews.dlq.v1"
    auto_offset_reset: Literal["earliest", "latest", "error"] = "earliest"

    mlflow_tracking_uri: AnyHttpUrl = "http://localhost:5000"
    model_name: NonBlankString = "rubert-sentiment"
    model_version: PositiveInt
    device: Literal["auto", "cpu", "cuda"] = "auto"

    batch_size: PositiveInt = 16
    batch_timeout_ms: PositiveInt = 200
    poll_timeout_seconds: PositiveFloat = 1.0
    delivery_timeout_seconds: PositiveFloat = 30.0

    @property
    def runtime_device(self) -> str | None:
        return None if self.device == "auto" else self.device

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

    def producer_config(self) -> dict:
        """Build an idempotent Kafka producer configuration."""

        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "client.id": self.kafka_client_id,


            "enable.idempotence": True,
            "acks": "all",
        }
