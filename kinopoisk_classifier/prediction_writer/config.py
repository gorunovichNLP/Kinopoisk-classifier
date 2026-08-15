"""Настройки Kafka Prediction Writer.

Архитектурный паттерн:
- Configuration Object: связанные настройки собраны в одном проверяемом объекте.

GoF-паттерны в этом файле не используются.
"""

# Literal ограничивает строку несколькими явно разрешёнными значениями.
from typing import Literal

# PositiveFloat не разрешает ноль и отрицательные значения timeout.
# PositiveInt используется для целого timeout подключения к PostgreSQL.
# PostgresDsn проверяет формат строки подключения к PostgreSQL.
from pydantic import PositiveFloat, PositiveInt, PostgresDsn

# BaseSettings читает значения из environment и .env.local.
# SettingsConfigDict задаёт правила чтения этих значений.
from pydantic_settings import BaseSettings, SettingsConfigDict

# NonBlankString запрещает пустые строки и строки только из пробелов.
from kinopoisk_classifier.shared.schemas import NonBlankString


class PredictionWriterSettings(BaseSettings):
    """Настройки с префиксом ``PREDICTION_WRITER_``."""

    # Например, поле input_topic можно переопределить переменной
    # PREDICTION_WRITER_INPUT_TOPIC без изменения Python-кода.
    model_config = SettingsConfigDict(
        env_prefix="PREDICTION_WRITER_",
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Kafka содержит готовые PredictionEventV1 от inference worker.
    kafka_bootstrap_servers: NonBlankString = "localhost:9092"
    kafka_client_id: NonBlankString = "kinopoisk-prediction-writer"
    kafka_group_id: NonBlankString = "kinopoisk-prediction-writer-v1"
    input_topic: NonBlankString = "kinopoisk.predictions.v1"

    # При первом запуске новой consumer group читаем topic с самого начала.
    auto_offset_reset: Literal["earliest", "latest", "error"] = "earliest"

    # run_once будет ждать одно Kafka-сообщение не дольше этого времени.
    poll_timeout_seconds: PositiveFloat = 1.0

    # PostgreSQL запущен отдельным контейнером и опубликован на host-порту 5433.
    postgres_dsn: PostgresDsn = (
        "postgresql://kinopoisk:kinopoisk@localhost:5433/kinopoisk_predictions"
    )

    # Если PostgreSQL недоступен, ошибка подключения появится через 5 секунд.
    postgres_connect_timeout_seconds: PositiveInt = 5

    def consumer_config(self) -> dict:
        """Возвращает простой словарь настроек для confluent-kafka Consumer."""

        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "client.id": self.kafka_client_id,
            "group.id": self.kafka_group_id,
            "auto.offset.reset": self.auto_offset_reset,
            # Offset нельзя подтверждать до успешного COMMIT в PostgreSQL.
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            # Не читаем сообщения из незавершённых Kafka transactions.
            "isolation.level": "read_committed",
        }
