"""Конфигурация Mongo Review Producer из переменных окружения."""

from pydantic import MongoDsn, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from kinopoisk_classifier.shared.schemas import NonBlankString


class ReviewProducerSettings(BaseSettings):
    """Настройки с префиксом ``REVIEW_PRODUCER_``.

    Например, ``REVIEW_PRODUCER_BATCH_SIZE=50`` переопределит размер batch,
    не требуя изменений в коде. Значения по умолчанию подходят только для
    локального Docker Compose из ``infra/docker``.
    """

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_PRODUCER_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB — источник данных и место хранения технического checkpoint.
    mongo_uri: MongoDsn = (
        "mongodb://kinopoisk:kinopoisk@localhost:27017/kinopoisk?authSource=admin"
    )
    mongo_database: NonBlankString = "kinopoisk"
    reviews_collection: NonBlankString = "reviews"
    checkpoints_collection: NonBlankString = "pipeline_checkpoints"

    # ID версионируем вместе с алгоритмом чтения. Если когда-нибудь изменим
    # смысл cursor-а, новый Producer не должен случайно продолжить старый.
    checkpoint_id: NonBlankString = "reviews-to-kafka-v1"

    kafka_bootstrap_servers: NonBlankString = "localhost:9092"
    kafka_client_id: NonBlankString = "kinopoisk-review-producer"
    output_topic: NonBlankString = "kinopoisk.reviews.v1"

    batch_size: PositiveInt = 50
    poll_interval_seconds: PositiveFloat = 1.0
    delivery_timeout_seconds: PositiveFloat = 30.0
    mongo_server_selection_timeout_ms: PositiveInt = 5_000

    def producer_config(self) -> dict:
        """Настройки Kafka producer-а для будущего шага публикации.

        Idempotence защищает от дублей, вызванных внутренними сетевыми retry
        одного процесса. Сбой между Kafka acknowledgement и записью checkpoint
        всё ещё может дать повтор — поэтому весь pipeline остаётся at-least-once.
        """

        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "client.id": self.kafka_client_id,
            "enable.idempotence": True,
            "acks": "all",
        }
