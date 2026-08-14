"""Kafka consumer -> ModelRuntime -> Kafka producer."""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event
from typing import Callable

from confluent_kafka import (
    Consumer,
    KafkaError,
    KafkaException,
    Producer,
    TopicPartition,
)
from pydantic import ValidationError

from kinopoisk_classifier.inference.config import InferenceSettings
from kinopoisk_classifier.inference.runtime import ModelRuntime
from kinopoisk_classifier.shared.schemas import (
    DeadLetterEventV1,
    PredictionEventV1,
    ReviewEventV1,
    SentimentPrediction,
    make_prediction_id,
)


LOG = logging.getLogger(__name__)
JSON_HEADERS = [("content-type", b"application/json"), ("schema-version", b"1")]


class KafkaDeliveryError(RuntimeError):
    """Хотя бы одно сообщение не было подтверждено Kafka."""


@dataclass(frozen=True)
class _InboundRecord:
    message: object
    event: ReviewEventV1 | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class _OutgoingRecord:
    topic: str
    key: bytes | None
    value: bytes


class InferenceWorker:
    """Обрабатывает Kafka-сообщения с гарантией at-least-once.

    At-least-once означает: потерять обработанное сообщение мы не должны, но
    после сбоя можем обработать его повторно. Повтор безопасен благодаря
    детерминированному prediction_id и будущему ON CONFLICT в PostgreSQL writer.
    """

    def __init__(
        self,
        settings: InferenceSettings,
        runtime: ModelRuntime,
        *,
        consumer=None,
        producer=None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.consumer = consumer or Consumer(settings.consumer_config())
        self.producer = producer or Producer(settings.producer_config())
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def run(self, stop_event: Event | None = None) -> None:
        """Запускает цикл до Ctrl+C или установки stop_event."""

        stop_event = stop_event or Event()
        self.consumer.subscribe([self.settings.input_topic])
        LOG.info("subscribed to %s", self.settings.input_topic)

        try:
            while not stop_event.is_set():
                self.run_once()
        except KeyboardInterrupt:
            LOG.info("shutdown requested")
        finally:
            self.producer.flush(self.settings.delivery_timeout_seconds)
            self.consumer.close()

    def run_once(self) -> int:
        """Собирает, публикует и подтверждает один батч.

        Это удобная единица не только рабочего цикла, но и тестирования: один
        вызов содержит полный путь consume -> inference -> produce -> commit.
        """

        records = self._poll_batch()
        if not records:
            return 0

        outgoing = self._build_outgoing(records)
        self._publish_and_wait(outgoing)

        # Commit выполняется только когда Kafka подтвердила все predictions и
        # DLQ этого батча. Сбой раньше оставит offsets на повторную обработку.
        self._commit(records)
        return len(records)

    def _poll_batch(self) -> list[_InboundRecord]:
        """Читает до batch_size сообщений, но не ждёт дольше batch_timeout."""

        records: list[_InboundRecord] = []
        deadline: float | None = None

        while len(records) < self.settings.batch_size:
            timeout = self.settings.poll_timeout_seconds
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout == 0.0:
                    break

            message = self.consumer.poll(timeout)
            if message is None:
                break
            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(message.error())

            if deadline is None:
                # Таймер стартует с первого сообщения. Поэтому при маленьком
                # потоке один отзыв не будет бесконечно ждать полного батча.
                deadline = time.monotonic() + self.settings.batch_timeout_ms / 1000
            records.append(self._parse(message))

        return records

    def _parse(self, message) -> _InboundRecord:
        """Преобразует Kafka bytes в ReviewEventV1 или сохраняет ошибку для DLQ."""

        try:
            if message.value() is None:
                raise ValueError("Kafka message value is null")

            event = ReviewEventV1.model_validate_json(message.value())
            # Одинаковый review_id в key гарантирует попадание событий одного
            # отзыва в одну partition. Проверяем key, а не доверяем producer-у.
            key = self._decode_key(message.key())
            if key != event.review_id:
                raise ValueError(
                    f"Kafka key {key!r} does not match review_id {event.review_id!r}"
                )
            return _InboundRecord(message=message, event=event)
        except (ValidationError, ValueError, UnicodeDecodeError) as exc:
            return _InboundRecord(message=message, error=exc)

    @staticmethod
    def _decode_key(key) -> str | None:
        if key is None:
            return None
        if isinstance(key, bytes):
            return key.decode("utf-8")
        return str(key)

    def _build_outgoing(self, records: list[_InboundRecord]) -> list[_OutgoingRecord]:
        """Строит predictions для валидных записей и DLQ для невалидных."""

        valid_records = [record for record in records if record.event is not None]
        predictions: list[SentimentPrediction] = []
        if valid_records:
            predictions = self.runtime.predict_batch(
                [record.event.text for record in valid_records]
            )

        if len(predictions) != len(valid_records):
            raise RuntimeError(
                "ModelRuntime returned a different number of predictions than inputs"
            )
        prediction_iterator = iter(predictions)
        now = self._clock()

        outgoing = []
        for record in records:
            if record.event is not None:
                event = self._prediction_event(
                    record.event,
                    next(prediction_iterator),
                    now,
                )
                outgoing.append(
                    _OutgoingRecord(
                        topic=self.settings.output_topic,
                        key=event.review_id.encode("utf-8"),
                        value=event.model_dump_json(exclude_none=True).encode("utf-8"),
                    )
                )
            else:
                outgoing.append(self._dead_letter(record, now))
        return outgoing

    def _prediction_event(
        self,
        review: ReviewEventV1,
        prediction: SentimentPrediction,
        predicted_at: datetime,
    ) -> PredictionEventV1:
        metadata = self.runtime.metadata
        return PredictionEventV1(
            schema_version=1,
            prediction_id=make_prediction_id(
                review.event_id,
                metadata.name,
                metadata.version,
            ),
            source_event_id=review.event_id,
            review_id=review.review_id,
            movie_id=review.movie_id,
            source_created_at=review.source_created_at,
            sentiment=prediction.sentiment,
            label_id=prediction.label_id,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            model=metadata,
            predicted_at=predicted_at,
        )

    def _dead_letter(
        self,
        record: _InboundRecord,
        failed_at: datetime,
    ) -> _OutgoingRecord:
        message = record.message
        error = record.error or RuntimeError("unknown validation error")
        key = self._as_bytes(message.key())
        payload = self._as_bytes(message.value()) or b""

        # Kafka value может быть любым набором байтов, не обязательно UTF-8.
        # Base64 сохраняет исходное сообщение без потерь внутри JSON DLQ-event.
        event = DeadLetterEventV1(
            schema_version=1,
            worker="sentiment-inference",
            source_topic=message.topic(),
            source_partition=message.partition(),
            source_offset=message.offset(),
            key_base64=base64.b64encode(key).decode("ascii") if key else None,
            payload_base64=base64.b64encode(payload).decode("ascii"),
            error_type=type(error).__name__,
            error_message=str(error) or repr(error),
            failed_at=failed_at,
        )
        return _OutgoingRecord(
            topic=self.settings.dlq_topic,
            key=key,
            value=event.model_dump_json(exclude_none=True).encode("utf-8"),
        )

    @staticmethod
    def _as_bytes(value) -> bytes | None:
        if value is None or isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")

    def _publish_and_wait(self, records: list[_OutgoingRecord]) -> None:
        """Публикует весь батч и ждёт delivery callback от Kafka."""

        errors = []

        def on_delivery(error, _message):
            if error is not None:
                errors.append(error)

        for record in records:
            # produce только кладёт record во внутреннюю очередь librdkafka.
            # Реальное подтверждение приходит позже через on_delivery.
            self.producer.produce(
                record.topic,
                key=record.key,
                value=record.value,
                headers=JSON_HEADERS,
                on_delivery=on_delivery,
            )
            self.producer.poll(0)

        # flush блокируется до подтверждения всех records или до timeout.
        remaining = self.producer.flush(self.settings.delivery_timeout_seconds)
        if remaining or errors:
            details = errors[0] if errors else f"{remaining} undelivered message(s)"
            raise KafkaDeliveryError(f"Kafka delivery failed: {details}")

    def _commit(self, records: list[_InboundRecord]) -> None:
        """Коммитит следующую позицию после последнего сообщения partition."""

        highest_offsets: dict[tuple[str, int], int] = {}
        for record in records:
            message = record.message
            partition = (message.topic(), message.partition())

            # Kafka commit хранит offset СЛЕДУЮЩЕГО сообщения. Если успешно
            # обработан offset=7, сохраняем 8. После рестарта consumer начнёт с 8.
            highest_offsets[partition] = max(
                highest_offsets.get(partition, 0),
                message.offset() + 1,
            )

        offsets = [
            TopicPartition(topic, partition, offset)
            for (topic, partition), offset in highest_offsets.items()
        ]
        self.consumer.commit(offsets=offsets, asynchronous=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = InferenceSettings()
    runtime = ModelRuntime.from_registry(
        tracking_uri=str(settings.mlflow_tracking_uri),
        model_name=settings.model_name,
        model_version=str(settings.model_version),
        device=settings.runtime_device,
        batch_size=settings.batch_size,
    )
    InferenceWorker(settings, runtime).run()


if __name__ == "__main__":
    main()
