"""Рабочий цикл Kafka → PostgreSQL Prediction Writer.

Паттерны проектирования:
- Mediator (GoF): PredictionWriter задаёт порядок работы Kafka Consumer и
  PostgreSQL Repository, которые ничего не знают друг о друге.
- Facade (GoF): run() запускает сервис, а run_once() выполняет одно сообщение.

Архитектурные приёмы:
- Application Service: класс реализует один прикладной сценарий системы.
- Dependency Injection: repository и необязательный consumer приходят снаружи.
- At-least-once Processing: Kafka offset подтверждается после PostgreSQL commit.
"""

# Event — потокобезопасный флаг для управляемой остановки цикла.
from threading import Event

# Consumer получает сообщения, а KafkaException сообщает об ошибке broker-а.
from confluent_kafka import Consumer, KafkaError, KafkaException

# Настройки содержат Kafka topic, group id и timeout poll.
from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings

# Repository выполняет INSERT и PostgreSQL commit.
from kinopoisk_classifier.prediction_writer.postgres import (
    PostgresPredictionRepository,
)

# Pydantic-модель проверяет JSON-контракт prediction-сообщения.
from kinopoisk_classifier.shared.schemas import PredictionEventV1


class PredictionWriter:
    """Постоянно читает predictions и надёжно сохраняет их в PostgreSQL."""

    def __init__(
        self,
        settings: PredictionWriterSettings,
        repository: PostgresPredictionRepository,
        *,
        consumer=None,
    ) -> None:
        self.settings = settings
        self.repository = repository

        # Настоящий Consumer создаётся в рабочем коде. Возможность передать
        # готовый consumer оставляем для будущих интеграционных сценариев.
        self._owns_consumer = consumer is None
        self.consumer = (
            Consumer(settings.consumer_config())
            if consumer is None
            else consumer
        )

        # Writer читает только topic с готовыми predictions.
        self.consumer.subscribe([settings.input_topic])

    def run(self, stop_event: Event | None = None) -> None:
        """Обрабатывает Kafka-сообщения, пока не запрошена остановка."""

        # При обычном запуске внешний Event не нужен: создаём его сами.
        # Интеграционный тест передаёт Event и вызывает set() для остановки.
        if stop_event is None:
            stop_event = Event()

        # После stop_event.set() новая итерация больше не начинается.
        while not stop_event.is_set():
            # run_once уже содержит весь безопасный порядок:
            # PostgreSQL commit выполняется раньше Kafka offset commit.
            self.run_once()

            # Дополнительный sleep здесь не нужен. Если сообщений нет,
            # consumer.poll() внутри run_once уже ждёт poll_timeout_seconds.

    def run_once(self) -> int:
        """Обрабатывает максимум одно сообщение и возвращает 0 или 1."""

        # poll ждёт сообщение не дольше заданного timeout.
        message = self.consumer.poll(self.settings.poll_timeout_seconds)

        # None означает, что за время ожидания нового сообщения не появилось.
        if message is None:
            return 0

        # Kafka может вернуть не данные, а служебное событие или ошибку.
        if message.error():
            # Конец partition не является поломкой: новых данных пока нет.
            if message.error().code() == KafkaError._PARTITION_EOF:
                return 0
            raise KafkaException(message.error())

        # Kafka value может быть null, но наш контракт этого не разрешает.
        if message.value() is None:
            raise ValueError("Kafka prediction message value is null")

        # Из bytes получаем и одновременно валидируем PredictionEventV1.
        prediction = PredictionEventV1.model_validate_json(message.value())

        # InferenceWorker использует review_id как Kafka key. Проверяем key,
        # чтобы случайно не сохранить противоречивое сообщение.
        message_key = self._decode_key(message.key())
        if message_key != prediction.review_id:
            raise ValueError(
                f"Kafka key {message_key!r} does not match "
                f"review_id {prediction.review_id!r}"
            )

        # save() возвращается только после PostgreSQL commit. Значение False
        # означает безопасный дубль, а не ошибку: строка уже была сохранена.
        self.repository.save(prediction)

        # Только теперь подтверждаем offset+1 именно этого Kafka-сообщения.
        # asynchronous=False заставляет дождаться ответа Kafka broker-а.
        self.consumer.commit(message=message, asynchronous=False)

        # Сообщение либо вставлено, либо уже существовало — оба случая успешны.
        return 1

    @staticmethod
    def _decode_key(key) -> str | None:
        """Преобразует Kafka key в обычную строку."""

        if key is None:
            return None
        if isinstance(key, bytes):
            return key.decode("utf-8")
        return str(key)

    def close(self) -> None:
        """Закрывает только Consumer, созданный самим Writer."""

        if self._owns_consumer:
            self.consumer.close()
