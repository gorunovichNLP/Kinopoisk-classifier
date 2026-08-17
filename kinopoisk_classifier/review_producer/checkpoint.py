"""Persist review producer progress in MongoDB."""


from collections.abc import Callable



from datetime import datetime, timezone


from bson import ObjectId



from pydantic import TypeAdapter


from pymongo import MongoClient


from kinopoisk_classifier.review_producer.config import ReviewProducerSettings


from kinopoisk_classifier.review_producer.schemas import ReviewProducerCheckpoint


from kinopoisk_classifier.shared.schemas import ObjectIdString




OBJECT_ID_ADAPTER = TypeAdapter(ObjectIdString)




class CheckpointConfigurationError(RuntimeError):
    """Raised when a checkpoint belongs to another pipeline."""



class CheckpointRegressionError(ValueError):
    """Raised when a checkpoint would move backwards."""




class MongoCheckpointStore:
    """MongoDB-backed persistent cursor store."""


    def __init__(

        self,

        settings: ReviewProducerSettings,

        *,


        client=None,


        clock: Callable[[], datetime] | None = None,

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


        database = self.client[settings.mongo_database]


        self.collection = database[settings.checkpoints_collection]



        self._clock = clock or (lambda: datetime.now(timezone.utc))


    def load(self) -> ReviewProducerCheckpoint | None:
        """Load and validate the current checkpoint."""



        document = self.collection.find_one(
            {"_id": self.settings.checkpoint_id}
        )


        if document is None:

            return None




        checkpoint = ReviewProducerCheckpoint.model_validate(document)



        self._validate_configuration(checkpoint)


        return checkpoint


    def save(self, last_review_id: str) -> ReviewProducerCheckpoint:
        """Persist a validated value transactionally."""



        normalized_id = OBJECT_ID_ADAPTER.validate_python(

            last_review_id,

            strict=True,
        )


        new_object_id = ObjectId(normalized_id)



        current = self.load()


        if current is not None:

            current_object_id = ObjectId(current.last_review_id)



            if new_object_id < current_object_id:

                raise CheckpointRegressionError(

                    "checkpoint cannot move backwards: "

                    f"current={current.last_review_id}, requested={normalized_id}"
                )


        self.collection.update_one(

            {"_id": self.settings.checkpoint_id},

            {

                "$set": {

                    "schema_version": 1,

                    "source_collection": self.settings.reviews_collection,

                    "target_topic": self.settings.output_topic,

                    "updated_at": self._clock(),
                },


                "$max": {"last_review_id": new_object_id},
            },

            upsert=True,
        )



        saved = self.load()



        if saved is None:
            raise RuntimeError("checkpoint was not found after MongoDB upsert")


        return saved


    def _validate_configuration(

        self,

        checkpoint: ReviewProducerCheckpoint,

    ) -> None:
        """Validate that the checkpoint belongs to this pipeline."""


        if checkpoint.source_collection != self.settings.reviews_collection:

            raise CheckpointConfigurationError(
                "checkpoint source_collection does not match settings: "
                f"{checkpoint.source_collection!r} != "
                f"{self.settings.reviews_collection!r}"
            )


        if checkpoint.target_topic != self.settings.output_topic:


            raise CheckpointConfigurationError(
                "checkpoint target_topic does not match settings: "
                f"{checkpoint.target_topic!r} != {self.settings.output_topic!r}"
            )


    def close(self) -> None:
        """Close resources owned by this instance."""


        if self._owns_client:

            self.client.close()
