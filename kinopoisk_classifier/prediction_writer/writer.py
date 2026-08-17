"""Consume Kafka predictions and persist them transactionally."""


from threading import Event


from confluent_kafka import Consumer, KafkaError, KafkaException


from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings


from kinopoisk_classifier.prediction_writer.postgres import (
    PostgresPredictionRepository,
)


from kinopoisk_classifier.shared.schemas import PredictionEventV1


class PredictionWriter:
    """Reliable Kafka-to-PostgreSQL application service."""

    def __init__(
        self,
        settings: PredictionWriterSettings,
        repository: PostgresPredictionRepository,
        *,
        consumer=None,
    ) -> None:
        self.settings = settings
        self.repository = repository



        self._owns_consumer = consumer is None
        self.consumer = (
            Consumer(settings.consumer_config())
            if consumer is None
            else consumer
        )


        self.consumer.subscribe([settings.input_topic])

    def run(self, stop_event: Event | None = None) -> None:
        """Run the service until a stop is requested."""



        if stop_event is None:
            stop_event = Event()


        while not stop_event.is_set():


            self.run_once()




    def run_once(self) -> int:
        """Process one available batch or message."""


        message = self.consumer.poll(self.settings.poll_timeout_seconds)


        if message is None:
            return 0


        if message.error():

            if message.error().code() == KafkaError._PARTITION_EOF:
                return 0
            raise KafkaException(message.error())


        if message.value() is None:
            raise ValueError("Kafka prediction message value is null")


        prediction = PredictionEventV1.model_validate_json(message.value())



        message_key = self._decode_key(message.key())
        if message_key != prediction.review_id:
            raise ValueError(
                f"Kafka key {message_key!r} does not match "
                f"review_id {prediction.review_id!r}"
            )



        self.repository.save(prediction)



        self.consumer.commit(message=message, asynchronous=False)


        return 1

    @staticmethod
    def _decode_key(key) -> str | None:
        """Decode a Kafka key as text."""

        if key is None:
            return None
        if isinstance(key, bytes):
            return key.decode("utf-8")
        return str(key)

    def close(self) -> None:
        """Close resources owned by this instance."""

        if self._owns_consumer:
            self.consumer.close()
