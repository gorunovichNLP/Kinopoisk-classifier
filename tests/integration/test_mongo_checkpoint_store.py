"""Integration tests for the MongoDB checkpoint store."""


import os


import unittest


import uuid


from bson import ObjectId


from pymongo import MongoClient



from kinopoisk_classifier.review_producer.checkpoint import (
    CheckpointRegressionError,
    MongoCheckpointStore,
)


from kinopoisk_classifier.review_producer.config import ReviewProducerSettings




@unittest.skipUnless(

    os.getenv("RUN_MONGO_INTEGRATION") == "1",

    "set RUN_MONGO_INTEGRATION=1 to use the Docker MongoDB",
)

class MongoCheckpointStoreIntegrationTest(unittest.TestCase):

    def test_saves_loads_and_advances_checkpoint(self):

        collection_name = f"checkpoints_integration_{uuid.uuid4().hex}"


        settings = ReviewProducerSettings(

            checkpoints_collection=collection_name,

            checkpoint_id=f"checkpoint-integration-{uuid.uuid4().hex}",

            _env_file=None,
        )



        seed_client = MongoClient(str(settings.mongo_uri), tz_aware=True)


        store = MongoCheckpointStore(settings)


        first_id = "66c0f12a9d2b6e41f1701201"
        second_id = "66c0f12a9d2b6e41f1701202"


        try:

            self.assertIsNone(store.load())


            store.save(first_id)


            saved = store.save(second_id)


            self.assertEqual(saved.last_review_id, second_id)


            database = seed_client[settings.mongo_database]


            collection = database[collection_name]


            raw = collection.find_one({"_id": settings.checkpoint_id})



            self.assertIsInstance(raw["last_review_id"], ObjectId)


            with self.assertRaises(CheckpointRegressionError):

                store.save(first_id)


        finally:

            store.close()



            seed_client[settings.mongo_database].drop_collection(collection_name)


            seed_client.close()



if __name__ == "__main__":

    unittest.main()
