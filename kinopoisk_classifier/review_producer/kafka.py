"""Преобразование MongoDB-отзывов в события и их публикация в Kafka."""

# Callable описывает функцию, которую можно вызвать.
# Sequence означает последовательность: например, list или tuple.
from collections.abc import Callable, Sequence

# datetime нужен для времени создания события.
# timezone.utc помогает всегда записывать время в UTC.
from datetime import datetime, timezone

# Producer — Kafka-клиент, который отправляет сообщения в topic.
from confluent_kafka import Producer

# В settings лежат адрес Kafka, название topic и timeout доставки.
from kinopoisk_classifier.review_producer.config import ReviewProducerSettings

# MongoReview — отзыв, который мы прочитали из MongoDB.
# ReviewEventV1 — событие, которое мы должны отправить в Kafka.
# make_review_event_id создаёт стабильный event_id из review_id.
from kinopoisk_classifier.shared.schemas import (
    MongoReview,
    ReviewEventV1,
    make_review_event_id,
)


# Kafka message состоит не только из key и value, но ещё может иметь headers.
# content-type говорит, что внутри value находится JSON.
# schema-version говорит, что мы используем первую версию контракта события.
# Значения headers передаём как bytes, поэтому перед строками стоит буква b.
JSON_HEADERS = [
    ("content-type", b"application/json"),
    ("schema-version", b"1"),
]


# Создаём собственный тип ошибки, чтобы вызывающий код мог отличить
# проблему доставки в Kafka от ошибки MongoDB или Pydantic-валидации.
class KafkaDeliveryError(RuntimeError):
    """Kafka не подтвердила доставку хотя бы одного события из batch."""


# Эта функция отвечает только за преобразование одной модели в другую.
# Она ничего не читает из MongoDB и ничего не отправляет в Kafka.
def review_to_event(
    # review — уже проверенный через Pydantic отзыв из MongoDB.
    review: MongoReview,
    # Звёздочка запрещает передавать emitted_at без имени аргумента.
    # Нужно писать emitted_at=..., так вызов легче читать.
    *,
    # emitted_at — момент, когда Producer создал Kafka-событие.
    emitted_at: datetime,
# Стрелка показывает, что функция возвращает ReviewEventV1.
) -> ReviewEventV1:
    """Строит внешний Kafka-контракт из внутренней MongoDB-модели.

    Преобразование держим в отдельной чистой функции: его можно проверить без
    запущенных MongoDB и Kafka, а правила маппинга полей видны в одном месте.
    """

    # Создание ReviewEventV1 одновременно запускает Pydantic-валидацию.
    return ReviewEventV1(
        # Указываем версию структуры события.
        schema_version=1,
        # event_id детерминирован: один review_id всегда даёт один event_id.
        event_id=make_review_event_id(review.review_id),
        # Идентификатор отзыва переносим из MongoDB без изменений.
        review_id=review.review_id,
        # Идентификатора фильма в MongoDB может не быть, тогда здесь будет None.
        movie_id=review.movie_id,
        # Текст отзыва — данные, которые затем получит inference worker.
        text=review.text,
        # Это время создания исходного отзыва, а не Kafka-события.
        source_created_at=review.created_at,
        # Это время создания именно Kafka-события.
        emitted_at=emitted_at,
    )


# Класс объединяет настройки Kafka, Kafka Producer и операцию публикации batch.
class KafkaReviewPublisher:
    """Публикует валидированные ``ReviewEventV1`` в Kafka.

    ``produce()`` у confluent-kafka асинхронный: вызов только помещает record
    во внутреннюю очередь клиента. Поэтому успешным batch считается лишь после
    ``flush()`` и delivery callback без ошибок от broker-а.
    """

    # __init__ вызывается при создании KafkaReviewPublisher(...).
    def __init__(
        # self — текущий экземпляр KafkaReviewPublisher.
        self,
        # settings содержат конфигурацию нашего Review Producer.
        settings: ReviewProducerSettings,
        # После звёздочки аргументы можно передавать только по имени.
        *,
        # В обычной работе producer не передаём — класс создаст настоящий.
        # В тестах сюда передаём fake producer, чтобы не запускать Kafka.
        producer=None,
        # clock — функция, возвращающая текущее время.
        # Callable[[], datetime] читается так: «без аргументов, вернёт datetime».
        clock: Callable[[], datetime] | None = None,
    # __init__ ничего не возвращает, поэтому его тип результата — None.
    ) -> None:
        # Сохраняем настройки внутри объекта, чтобы использовать их позднее.
        self.settings = settings

        # Если готовый producer не передан, создаём настоящий Kafka Producer.
        # producer_config() возвращает словарь настроек для confluent-kafka.
        self.producer = (
            Producer(settings.producer_config())
            if producer is None
            else producer
        )

        # В рабочем коде используем текущее UTC-время.
        # В тесте можно передать clock с фиксированным временем.
        # lambda — маленькая функция без имени.
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # Метод получает несколько MongoReview и публикует их как один batch.
    def publish_batch(
        # self — текущий publisher.
        self,
        # reviews может быть списком или другой последовательностью MongoReview.
        reviews: Sequence[MongoReview],
    # После успеха возвращаем список реально созданных событий.
    ) -> list[ReviewEventV1]:
        """Публикует batch и возвращает события после подтверждения Kafka.

        Сначала строим весь batch. Если хотя бы один объект нарушает Pydantic-
        контракт, ошибка возникнет до первой отправки и частичного batch не будет.
        """

        # Один раз получаем время, поэтому у всех событий batch одинаковый emitted_at.
        emitted_at = self._clock()

        # Преобразуем каждый MongoReview в валидный ReviewEventV1.
        # Это происходит до отправки первого сообщения в Kafka.
        events = [
            review_to_event(review, emitted_at=emitted_at)
            for review in reviews
        ]

        # Пустой batch не требует обращения к Kafka.
        if not events:
            # Сразу возвращаем пустой список.
            return []

        # Сюда callback будет складывать ошибки доставки отдельных сообщений.
        delivery_errors = []

        # Kafka вызывает эту функцию, когда доставка конкретного сообщения закончилась.
        def on_delivery(error, _message) -> None:
            # error равен None при успехе и содержит ошибку при неудаче.
            if error is not None:
                # Сохраняем ошибку, чтобы проверить её после завершения всего batch.
                delivery_errors.append(error)

        # По очереди ставим каждое событие во внутреннюю очередь Producer.
        for event in events:
            # produce() работает асинхронно: он не ждёт ответа Kafka broker-а.
            self.producer.produce(
                # Все отзывы публикуем в topic из настроек.
                self.settings.output_topic,
                # Kafka распределяет сообщения по partitions на основе key.
                # encode превращает Python-строку в bytes для Kafka.
                key=event.review_id.encode("utf-8"),
                # Pydantic превращает модель в JSON-строку.
                # exclude_none=True не записывает отсутствующие необязательные поля.
                # Затем encode превращает JSON-строку в bytes.
                value=event.model_dump_json(exclude_none=True).encode("utf-8"),
                # Добавляем описание формата сообщения.
                headers=JSON_HEADERS,
                # Передаём функцию, которую Kafka-клиент вызовет после доставки.
                on_delivery=on_delivery,
            )

            # poll(0) обрабатывает готовые callbacks и совсем не ждёт.
            # Число 0 означает timeout в ноль секунд.
            self.producer.poll(0)

        # flush ждёт, пока очередь Producer опустеет или закончится timeout.
        # Результат — количество сообщений, которые всё ещё остались в очереди.
        remaining = self.producer.flush(
            self.settings.delivery_timeout_seconds
        )

        # Ошибка есть, если сообщения остались или callback сообщил о проблеме.
        if remaining or delivery_errors:
            # Если callback дал конкретную ошибку, показываем именно её.
            # Иначе сообщаем количество сообщений, оставшихся после timeout.
            details = (
                delivery_errors[0]
                if delivery_errors
                else f"{remaining} undelivered message(s)"
            )

            # Не позволяем будущему checkpoint считать этот batch успешным.
            raise KafkaDeliveryError(f"Kafka delivery failed: {details}")

        # До этой строки доходим только после подтверждения всего batch.
        # Возвращённые события позже помогут определить последний review_id.
        return events
