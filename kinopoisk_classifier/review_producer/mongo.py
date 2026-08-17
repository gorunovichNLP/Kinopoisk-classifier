"""Read immutable MongoDB reviews in ordered batches."""

from bson import ObjectId
from pydantic import TypeAdapter
from pymongo import ASCENDING, MongoClient

from kinopoisk_classifier.review_producer.config import ReviewProducerSettings
from kinopoisk_classifier.shared.schemas import MongoReview, ObjectIdString


OBJECT_ID_ADAPTER = TypeAdapter(ObjectIdString)


class MongoReviewReader:
    """Ordered reader for immutable MongoDB reviews."""

    def __init__(
        self,
        settings: ReviewProducerSettings,
        *,
        client=None,
    ) -> None:
        self.settings = settings
        self._owns_client = client is None




        if client is None:
            self.client = MongoClient(
                str(settings.mongo_uri),
                tz_aware=True,
                serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
            )
        else:


            self.client = client
        self.collection = self.client[settings.mongo_database][
            settings.reviews_collection
        ]

    def read_batch(self, after_review_id: str | None = None) -> list[MongoReview]:
        """Read the next ordered batch after an optional cursor."""

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
        """Verify the backing service connection."""

        self.client.admin.command("ping")

    def close(self) -> None:
        """Close resources owned by this instance."""

        if self._owns_client:
            self.client.close()
