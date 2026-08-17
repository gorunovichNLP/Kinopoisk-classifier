"""Executable composition root for the review producer service."""


import logging


from threading import Event


from pymongo import MongoClient


from kinopoisk_classifier.review_producer.checkpoint import MongoCheckpointStore


from kinopoisk_classifier.review_producer.config import ReviewProducerSettings


from kinopoisk_classifier.review_producer.kafka import KafkaReviewPublisher


from kinopoisk_classifier.review_producer.mongo import MongoReviewReader


from kinopoisk_classifier.review_producer.producer import ReviewProducer



LOG = logging.getLogger(__name__)



def main(stop_event: Event | None = None) -> None:
    """Run the command-line entry point."""


    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )



    settings = ReviewProducerSettings()



    mongo_client = MongoClient(

        str(settings.mongo_uri),

        tz_aware=True,

        serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
    )



    publisher = None


    try:


        reader = MongoReviewReader(settings, client=mongo_client)


        checkpoint_store = MongoCheckpointStore(settings, client=mongo_client)


        publisher = KafkaReviewPublisher(settings)



        review_producer = ReviewProducer(
            settings,
            reader,
            publisher,
            checkpoint_store,
        )



        reader.ping()


        LOG.info(
            "starting ReviewProducer: collection=%s topic=%s checkpoint=%s",
            settings.reviews_collection,
            settings.output_topic,
            settings.checkpoint_id,
        )




        review_producer.run(stop_event)


    except KeyboardInterrupt:

        LOG.info("ReviewProducer shutdown requested by user")


    finally:

        if publisher is not None:


            remaining = publisher.producer.flush(
                settings.delivery_timeout_seconds
            )




            if remaining:
                LOG.warning(
                    "Kafka producer stopped with %s undelivered message(s)",
                    remaining,
                )


        mongo_client.close()


        LOG.info("ReviewProducer stopped")



if __name__ == "__main__":

    # python -m kinopoisk_classifier.review_producer
    main()
