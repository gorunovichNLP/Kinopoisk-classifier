"""Чтение и сохранение прогресса Review Producer в MongoDB."""

# Callable нужен для типа функции clock, которая возвращает текущее время.
from collections.abc import Callable

# datetime описывает дату и время.
# timezone.utc позволяет сохранять время в единой временной зоне UTC.
from datetime import datetime, timezone

# ObjectId — специальный BSON-тип идентификатора MongoDB.
from bson import ObjectId

# TypeAdapter позволяет применять правила Pydantic к отдельному значению,
# даже если это значение не является целой Pydantic-моделью.
from pydantic import TypeAdapter

# MongoClient создаёт подключение к MongoDB.
from pymongo import MongoClient

# В settings находятся адрес MongoDB, имена базы и коллекций, checkpoint_id.
from kinopoisk_classifier.review_producer.config import ReviewProducerSettings

# ReviewProducerCheckpoint описывает правильную структуру checkpoint-документа.
from kinopoisk_classifier.review_producer.schemas import ReviewProducerCheckpoint

# ObjectIdString содержит Pydantic-правила для строки из 24 hex-символов.
from kinopoisk_classifier.shared.schemas import ObjectIdString


# Подготавливаем отдельный валидатор для last_review_id.
# Например, строка "hello" не пройдёт эту проверку как Mongo ObjectId.
OBJECT_ID_ADAPTER = TypeAdapter(ObjectIdString)


# Отдельный тип ошибки помогает понять, что структура checkpoint правильная,
# но он был создан для другой Mongo-коллекции или другого Kafka topic.
class CheckpointConfigurationError(RuntimeError):
    """Сохранённый checkpoint относится к другой коллекции или topic."""


# Эта ошибка означает попытку заменить более новый указатель более старым.
class CheckpointRegressionError(ValueError):
    """Попытка передвинуть checkpoint назад к уже обработанному отзыву."""


# Store — небольшой класс, отвечающий только за checkpoint-документ.
# Он не читает отзывы и не отправляет сообщения в Kafka.
class MongoCheckpointStore:
    """Хранит один именованный checkpoint в технической MongoDB-коллекции.

    Отзывы и checkpoint лежат в одной базе, но в разных коллекциях. Store не
    читает сами отзывы и не работает с Kafka — он отвечает только за прогресс.
    """

    # __init__ вызывается при создании MongoCheckpointStore(...).
    def __init__(
        # self — создаваемый экземпляр MongoCheckpointStore.
        self,
        # settings — проверенные Pydantic-настройки Review Producer.
        settings: ReviewProducerSettings,
        # После звёздочки аргументы можно передавать только по имени.
        *,
        # В рабочем коде client не передаётся, и мы создаём настоящий MongoClient.
        # В тестах можно передать fake client или заранее созданный MongoClient.
        client=None,
        # clock — функция без аргументов, которая возвращает datetime.
        # Благодаря этому тест может использовать фиксированное время.
        clock: Callable[[], datetime] | None = None,
    # Конструктор только настраивает объект и ничего не возвращает.
    ) -> None:
        # Сохраняем настройки внутри store для дальнейших методов.
        self.settings = settings

        # Запоминаем владельца client.
        # True означает: client был создан внутри этого store.
        self._owns_client = client is None

        # Если готовый client нам не передали, создаём настоящий MongoClient.
        if client is None:
            # Само создание MongoClient ещё не обязательно открывает соединение:
            # PyMongo подключается лениво при первой реальной операции.
            self.client = MongoClient(
                # MongoDsn — объект Pydantic, а PyMongo ожидает обычную строку.
                str(settings.mongo_uri),
                # Возвращаемые даты должны содержать информацию о timezone.
                tz_aware=True,
                # Не ждём MongoDB бесконечно, если сервер недоступен.
                serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
            )
        # Если client передали снаружи, используем именно его.
        else:
            self.client = client

        # Сначала по имени выбираем базу данных MongoDB.
        database = self.client[settings.mongo_database]

        # Затем внутри базы выбираем техническую коллекцию checkpoints.
        self.collection = database[settings.checkpoints_collection]

        # Если clock передан, сохраняем его.
        # Иначе создаём функцию, которая возвращает текущее UTC-время.
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # Метод загрузки ничего не изменяет в MongoDB.
    def load(self) -> ReviewProducerCheckpoint | None:
        """Возвращает текущий checkpoint или ``None`` при первом запуске."""

        # Ищем ровно один документ по его MongoDB-полю _id.
        # checkpoint_id берём из настроек, например "reviews-to-kafka-v1".
        document = self.collection.find_one(
            {"_id": self.settings.checkpoint_id}
        )

        # find_one возвращает None, если подходящего документа нет.
        if document is None:
            # Отсутствие checkpoint означает первый запуск Producer.
            return None

        # Превращаем обычный словарь PyMongo в Pydantic-модель.
        # Здесь проверятся обязательные поля, типы и запрет лишних полей.
        # BSON ObjectId из last_review_id станет обычной строкой.
        checkpoint = ReviewProducerCheckpoint.model_validate(document)

        # Проверяем не только структуру, но и смысл документа:
        # он должен относиться к нашей коллекции отзывов и нашему Kafka topic.
        self._validate_configuration(checkpoint)

        # Возвращаем наружу уже проверенную неизменяемую Pydantic-модель.
        return checkpoint

    # Метод сохранения получает id последнего подтверждённого Kafka отзыва.
    def save(self, last_review_id: str) -> ReviewProducerCheckpoint:
        """Атомарно продвигает checkpoint и возвращает сохранённый документ.

        Вызывать этот метод можно только после Kafka acknowledgement всего
        batch. MongoDB-оператор ``$max`` не позволяет указателю стать меньше,
        даже если два процесса случайно попытаются обновить документ вместе.
        """

        # Явно запускаем Pydantic-валидацию перед обращением к MongoDB.
        # На выходе получим нормализованную строку в нижнем регистре.
        normalized_id = OBJECT_ID_ADAPTER.validate_python(
            # Проверяем значение, которое передал вызывающий код.
            last_review_id,
            # strict=True запрещает нежелательные неявные преобразования типов.
            strict=True,
        )

        # Для хранения в BSON превращаем строку в настоящий ObjectId.
        new_object_id = ObjectId(normalized_id)

        # Загружаем текущий checkpoint перед обновлением.
        # Заодно load проверит Pydantic-контракт и конфигурацию документа.
        current = self.load()

        # Проверку выполняем только тогда, когда checkpoint уже существует.
        if current is not None:
            # В Pydantic-модели id хранится строкой, поэтому снова создаём ObjectId.
            current_object_id = ObjectId(current.last_review_id)

            # MongoReviewReader читает документы по возрастанию ObjectId.
            # Следовательно, новый checkpoint не должен быть меньше текущего.
            if new_object_id < current_object_id:
                # Не выполняем update и явно сообщаем о логической ошибке.
                raise CheckpointRegressionError(
                    # Первая часть объясняет тип ошибки.
                    "checkpoint cannot move backwards: "
                    # Вторая часть показывает оба значения для диагностики.
                    f"current={current.last_review_id}, requested={normalized_id}"
                )

        # update_one изменяет один документ, найденный по checkpoint_id.
        self.collection.update_one(
            # Первый аргумент — условие поиска документа.
            {"_id": self.settings.checkpoint_id},
            # Второй аргумент — MongoDB-операторы обновления.
            {
                # $set записывает поля независимо от их предыдущих значений.
                "$set": {
                    # Версия позволяет позднее безопасно менять структуру документа.
                    "schema_version": 1,
                    # Запоминаем, какую Mongo-коллекцию читает этот checkpoint.
                    "source_collection": self.settings.reviews_collection,
                    # Запоминаем, в какой Kafka topic публикуются отзывы.
                    "target_topic": self.settings.output_topic,
                    # Фиксируем время этого обновления в UTC.
                    "updated_at": self._clock(),
                },
                # $max меняет поле только тогда, когда новое значение больше.
                # Это атомарная дополнительная защита от движения назад.
                "$max": {"last_review_id": new_object_id},
            },
            # Если документа ещё нет, MongoDB автоматически создаст его.
            upsert=True,
        )

        # После записи снова читаем checkpoint из MongoDB.
        # Так мы возвращаем именно сохранённое значение, а не локальное предположение.
        saved = self.load()

        # После успешного acknowledged upsert документ должен существовать.
        # Проверка оставлена как защита от неожиданного поведения client/fake client.
        if saved is None:
            raise RuntimeError("checkpoint was not found after MongoDB upsert")

        # Возвращаем повторно провалидированный checkpoint вызывающему коду.
        return saved

    # Начальный символ подчёркивания означает внутренний метод класса.
    def _validate_configuration(
        # self даёт доступ к settings текущего store.
        self,
        # checkpoint — уже структурно проверенная Pydantic-модель.
        checkpoint: ReviewProducerCheckpoint,
    # Метод либо успешно заканчивается, либо выбрасывает исключение.
    ) -> None:
        """Не даёт продолжить чужой checkpoint с тем же ``_id``."""

        # Сравниваем сохранённое имя исходной коллекции с текущими настройками.
        if checkpoint.source_collection != self.settings.reviews_collection:
            # Несовпадение опасно: last_review_id мог относиться к другим данным.
            raise CheckpointConfigurationError(
                "checkpoint source_collection does not match settings: "
                f"{checkpoint.source_collection!r} != "
                f"{self.settings.reviews_collection!r}"
            )

        # Аналогично проверяем Kafka topic, для которого сохранялся прогресс.
        if checkpoint.target_topic != self.settings.output_topic:
            # Нельзя считать отзыв опубликованным в новом topic только потому,
            # что он когда-то был опубликован в другом topic.
            raise CheckpointConfigurationError(
                "checkpoint target_topic does not match settings: "
                f"{checkpoint.target_topic!r} != {self.settings.output_topic!r}"
            )

    # close освобождает ресурсы созданного нами MongoClient.
    def close(self) -> None:
        """Закрывает только тот MongoClient, который store создал сам."""

        # Чужой client закрывать нельзя: его может использовать другой компонент.
        if self._owns_client:
            # Закрываем сетевые соединения внутреннего MongoClient.
            self.client.close()
