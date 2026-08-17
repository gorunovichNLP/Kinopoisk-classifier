"""Integration tests for ordered MongoDB review reads."""

import os
import unittest
import uuid
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import MongoClient

from kinopoisk_classifier.review_producer.config import ReviewProducerSettings
from kinopoisk_classifier.review_producer.mongo import MongoReviewReader


@unittest.skipUnless(
    os.getenv("RUN_MONGO_INTEGRATION") == "1",
    "set RUN_MONGO_INTEGRATION=1 to use the Docker MongoDB",
)
class MongoReviewReaderIntegrationTest(unittest.TestCase):
    def test_reads_ordered_pages_after_object_id(self):


        collection_name = f"reviews_integration_{uuid.uuid4().hex}"
        settings = ReviewProducerSettings(
            reviews_collection=collection_name,
            batch_size=2,
            _env_file=None,
        )
        seed_client = MongoClient(str(settings.mongo_uri), tz_aware=True)
        collection = seed_client[settings.mongo_database][collection_name]
        reader = MongoReviewReader(settings)

        review_ids = [
            ObjectId("66c0f12a9d2b6e41f1701201"),
            ObjectId("66c0f12a9d2b6e41f1701202"),
            ObjectId("66c0f12a9d2b6e41f1701203"),
        ]

        try:


            collection.insert_many(
                [
                    {
                        "_id": review_ids[2],
                        "text": "Third review",
                        "created_at": datetime.now(timezone.utc),
                    },
                    {
                        "_id": review_ids[0],
                        "text": "First review",
                        "created_at": datetime.now(timezone.utc),
                        "unused_source_field": "MongoReview ignores this field",
                    },
                    {
                        "_id": review_ids[1],
                        "text": "Second review",
                        "created_at": datetime.now(timezone.utc),
                    },
                ]
            )

            reader.ping()
            first_page = reader.read_batch()
            second_page = reader.read_batch(after_review_id=first_page[-1].review_id)

            self.assertEqual(
                [review.review_id for review in first_page],
                [str(review_ids[0]), str(review_ids[1])],
            )
            self.assertEqual(
                [review.review_id for review in second_page],
                [str(review_ids[2])],
            )
        finally:
            reader.close()

            seed_client[settings.mongo_database].drop_collection(collection_name)
            seed_client.close()


if __name__ == "__main__":
    unittest.main()
