"""Чтение неизменяемых отзывов из MongoDB небольшими batch-ами."""

from bson import ObjectId
from pydantic import TypeAdapter
from pymongo import ASCENDING, MongoClient

from kinopoisk_classifier.review_producer.config import ReviewProducerSettings
from kinopoisk_classifier.shared.schemas import MongoReview, ObjectIdString


OBJECT_ID_ADAPTER = TypeAdapter(ObjectIdString)


class MongoReviewReader:
    """Читает следующую страницу отзывов после заданного Mongo ObjectId.

    Reader пока ничего не публикует и не записывает в MongoDB. Его единственная
    задача на этом шаге — построить корректный запрос, отсортировать документы
    и провалидировать каждый результат через ``MongoReview``.
    """

    def __init__(
        self,
        settings: ReviewProducerSettings,
        *,
        client=None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None

        # MongoClient устанавливает соединение лениво: реальный сетевой запрос
        # произойдёт при первом find или command. tz_aware критичен для нашего
        # контракта — даты из BSON должны приходить как timezone-aware UTC.
        if client is None:
            self.client = MongoClient(
                str(settings.mongo_uri),
                tz_aware=True,
                serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
            )
        else:
            # Явная проверка `is None` важнее короткого `client or ...`:
            # объекты database-драйверов не обязаны поддерживать bool(client).
            self.client = client
        self.collection = self.client[settings.mongo_database][
            settings.reviews_collection
        ]

    def read_batch(self, after_review_id: str | None = None) -> list[MongoReview]:
        """Возвращает не больше ``batch_size`` отзывов по возрастанию ``_id``.

        ``after_review_id=None`` означает первый запуск и чтение с начала.
        Иначе запрос использует строгое ``$gt``: сам checkpoint-документ второй
        раз в batch не попадёт.
        """

        query = {}
        if after_review_id is not None:
            normalized_id = OBJECT_ID_ADAPTER.validate_python(
                after_review_id,
                strict=True,
            )
            query["_id"] = {"$gt": ObjectId(normalized_id)}

        cursor = (
            self.collection.find(query)
            .sort("_id", ASCENDING)
            .limit(self.settings.batch_size)
        )
        return [MongoReview.model_validate(document) for document in cursor]

    def ping(self) -> None:
        """Принудительно проверяет соединение, потому что MongoClient ленивый."""

        self.client.admin.command("ping")

    def close(self) -> None:
        """Закрывает только тот MongoClient, который reader создал сам."""

        if self._owns_client:
            self.client.close()
