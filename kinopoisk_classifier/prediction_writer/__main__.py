"""Executable composition root for the prediction writer service."""


import logging


from threading import Event


from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings


from kinopoisk_classifier.prediction_writer.postgres import (
    PostgresPredictionRepository,
)


from kinopoisk_classifier.prediction_writer.writer import PredictionWriter



LOG = logging.getLogger(__name__)


def main(stop_event: Event | None = None) -> None:
    """Run the command-line entry point."""


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


    settings = PredictionWriterSettings()



    repository = None
    writer = None

    try:

        repository = PostgresPredictionRepository(settings)


        repository.ping()


        writer = PredictionWriter(settings, repository)


        LOG.info(
            "starting PredictionWriter: topic=%s group=%s",
            settings.input_topic,
            settings.kafka_group_id,
        )



        writer.run(stop_event)


    except KeyboardInterrupt:
        LOG.info("PredictionWriter shutdown requested by user")


    finally:

        if writer is not None:
            writer.close()


        if repository is not None:
            repository.close()

        LOG.info("PredictionWriter stopped")



if __name__ == "__main__":

    main()
