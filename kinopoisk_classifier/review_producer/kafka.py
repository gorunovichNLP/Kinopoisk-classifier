"""Convert MongoDB reviews to events and publish them to Kafka."""



from collections.abc import Callable, Sequence



from datetime import datetime, timezone


from confluent_kafka import Producer


from kinopoisk_classifier.review_producer.config import ReviewProducerSettings




from kinopoisk_classifier.shared.schemas import (
    MongoReview,
    ReviewEventV1,
    make_review_event_id,
)






JSON_HEADERS = [
    ("content-type", b"application/json"),
    ("schema-version", b"1"),
]




class KafkaDeliveryError(RuntimeError):
    """Raised when Kafka does not acknowledge every record."""




def review_to_event(

    review: MongoReview,


    *,

    emitted_at: datetime,

) -> ReviewEventV1:
    """Convert a validated MongoDB review into a Kafka event."""


    return ReviewEventV1(

        schema_version=1,

        event_id=make_review_event_id(review.review_id),

        review_id=review.review_id,

        movie_id=review.movie_id,

        text=review.text,

        source_created_at=review.created_at,

        emitted_at=emitted_at,
    )



class KafkaReviewPublisher:
    """Publisher for validated review events."""


    def __init__(

        self,

        settings: ReviewProducerSettings,

        *,


        producer=None,


        clock: Callable[[], datetime] | None = None,

    ) -> None:

        self.settings = settings



        self.producer = (
            Producer(settings.producer_config())
            if producer is None
            else producer
        )




        self._clock = clock or (lambda: datetime.now(timezone.utc))


    def publish_batch(

        self,

        reviews: Sequence[MongoReview],

    ) -> list[ReviewEventV1]:
        """Publish a complete batch and wait for Kafka acknowledgement."""


        emitted_at = self._clock()



        events = [
            review_to_event(review, emitted_at=emitted_at)
            for review in reviews
        ]


        if not events:

            return []


        delivery_errors = []


        def on_delivery(error, _message) -> None:

            if error is not None:

                delivery_errors.append(error)


        for event in events:

            self.producer.produce(

                self.settings.output_topic,


                key=event.review_id.encode("utf-8"),



                value=event.model_dump_json(exclude_none=True).encode("utf-8"),

                headers=JSON_HEADERS,

                on_delivery=on_delivery,
            )



            self.producer.poll(0)



        remaining = self.producer.flush(
            self.settings.delivery_timeout_seconds
        )


        if remaining or delivery_errors:


            details = (
                delivery_errors[0]
                if delivery_errors
                else f"{remaining} undelivered message(s)"
            )


            raise KafkaDeliveryError(f"Kafka delivery failed: {details}")



        return events
