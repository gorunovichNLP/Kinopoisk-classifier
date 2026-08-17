"""Coordinate the reliable MongoDB-to-Kafka processing loop."""


from threading import Event


from kinopoisk_classifier.review_producer.checkpoint import MongoCheckpointStore


from kinopoisk_classifier.review_producer.config import ReviewProducerSettings


from kinopoisk_classifier.review_producer.kafka import KafkaReviewPublisher


from kinopoisk_classifier.review_producer.mongo import MongoReviewReader


class ReviewProducer:
    """Reliable MongoDB-to-Kafka application service."""

    def __init__(
        self,

        settings: ReviewProducerSettings,


        reader: MongoReviewReader,
        publisher: KafkaReviewPublisher,
        checkpoint_store: MongoCheckpointStore,
    ) -> None:

        self.settings = settings
        self.reader = reader
        self.publisher = publisher
        self.checkpoint_store = checkpoint_store

    def run(self, stop_event: Event | None = None) -> None:
        """Run the service until a stop is requested."""



        if stop_event is None:
            stop_event = Event()


        while not stop_event.is_set():

            processed_count = self.run_once()



            if processed_count == 0:

                stop_event.wait(self.settings.poll_interval_seconds)

    def run_once(self) -> int:
        """Process one available batch or message."""



        checkpoint = self.checkpoint_store.load()



        after_review_id = (
            checkpoint.last_review_id if checkpoint is not None else None
        )


        reviews = self.reader.read_batch(after_review_id=after_review_id)


        if not reviews:

            return 0




        self.publisher.publish_batch(reviews)



        self.checkpoint_store.save(reviews[-1].review_id)



        return len(reviews)
