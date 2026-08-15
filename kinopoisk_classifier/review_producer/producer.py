"""Одна итерация рабочего цикла MongoDB → Kafka.

Паттерны проектирования:
- Mediator (GoF): ReviewProducer координирует reader, publisher и checkpoint
  store, а эти компоненты ничего не знают друг о друге.
- Facade (GoF): внешний код запускает весь сценарий одним методом run_once().

Архитектурные приёмы:
- Application Service: класс реализует один прикладной сценарий системы.
- Dependency Injection: зависимости передаются через конструктор.
"""

# Store загружает и сохраняет позицию чтения MongoDB.
from kinopoisk_classifier.review_producer.checkpoint import MongoCheckpointStore

# Publisher превращает отзывы в события и ждёт подтверждения Kafka.
from kinopoisk_classifier.review_producer.kafka import KafkaReviewPublisher

# Reader читает следующую упорядоченную страницу отзывов из MongoDB.
from kinopoisk_classifier.review_producer.mongo import MongoReviewReader


class ReviewProducer:
    """Координирует одну безопасную итерацию MongoDB → Kafka → checkpoint."""

    def __init__(
        self,
        # Каждый объект отвечает только за одну операцию. ReviewProducer не
        # создаёт зависимости сам, поэтому его легко тестировать через fake-объекты.
        reader: MongoReviewReader,
        publisher: KafkaReviewPublisher,
        checkpoint_store: MongoCheckpointStore,
    ) -> None:
        # Сохраняем зависимости, чтобы использовать их при каждом run_once().
        self.reader = reader
        self.publisher = publisher
        self.checkpoint_store = checkpoint_store

    def run_once(self) -> int:
        """Обрабатывает не больше одного batch и возвращает число отзывов.

        Метод намеренно не содержит бесконечного цикла и sleep. Одна небольшая
        итерация проще для понимания и позволяет отдельно проверить критический
        порядок: publish должен полностью завершиться раньше checkpoint.save.
        """

        # При первом запуске load() вернёт None. После успешных публикаций здесь
        # будет id последнего отзыва, уже подтверждённого Kafka.
        checkpoint = self.checkpoint_store.load()

        # None означает «читать с самого начала коллекции reviews».
        # Иначе reader построит условие: _id > checkpoint.last_review_id.
        after_review_id = (
            checkpoint.last_review_id if checkpoint is not None else None
        )

        # Reader сортирует документы по _id и ограничивает результат batch_size.
        reviews = self.reader.read_batch(after_review_id=after_review_id)

        # Пустой список означает, что новых отзывов сейчас нет.
        if not reviews:
            # Kafka и checkpoint в этой ситуации трогать не нужно.
            return 0

        # publish_batch вернётся только после delivery acknowledgement всех
        # событий. При ошибке он выбросит исключение, и следующая строка не
        # выполнится — checkpoint останется на безопасной прежней позиции.
        self.publisher.publish_batch(reviews)

        # Весь batch подтверждён Kafka. Теперь можно запомнить последний id.
        # reviews отсортированы по возрастанию _id, поэтому берём элемент [-1].
        self.checkpoint_store.save(reviews[-1].review_id)

        # Число пригодится будущему run(): 0 означает отсутствие новых данных,
        # положительное значение — успешно обработанный batch.
        return len(reviews)
