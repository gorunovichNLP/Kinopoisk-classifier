"""Исполняемая точка входа MongoDB Review Producer.

Архитектурные паттерны:
- Composition Root: именно здесь создаются реальные инфраструктурные объекты
  и связываются в готовое приложение.
- Dependency Injection: созданные зависимости передаются в ReviewProducer.
- Shared Resource: reader и checkpoint store используют один MongoClient.

GoF Factory Method здесь не используется: функция main() напрямую создаёт
конкретные объекты и не переопределяется через наследование.
"""

# logging выводит информацию о запуске и корректной остановке сервиса.
import logging

# MongoClient создаёт и держит пул соединений с MongoDB.
from pymongo import MongoClient

# Store хранит id последнего отзыва, подтверждённого Kafka.
from kinopoisk_classifier.review_producer.checkpoint import MongoCheckpointStore

# Settings читает и валидирует переменные окружения REVIEW_PRODUCER_*.
from kinopoisk_classifier.review_producer.config import ReviewProducerSettings

# Publisher отправляет ReviewEventV1 в Kafka.
from kinopoisk_classifier.review_producer.kafka import KafkaReviewPublisher

# Reader читает следующие отзывы из MongoDB.
from kinopoisk_classifier.review_producer.mongo import MongoReviewReader

# ReviewProducer связывает reader, publisher и checkpoint store в рабочий цикл.
from kinopoisk_classifier.review_producer.producer import ReviewProducer


# Получаем logger с именем текущего Python-модуля.
LOG = logging.getLogger(__name__)


# main — единственная функция, необходимая для запуска приложения.
def main() -> None:
    """Создаёт зависимости, запускает ReviewProducer и освобождает ресурсы."""

    # Настраиваем корневой logger до первого сообщения приложения.
    logging.basicConfig(
        # INFO показывает важные этапы, но не засоряет вывод отладочными деталями.
        level=logging.INFO,
        # В каждой строке будут время, уровень, имя модуля и само сообщение.
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Pydantic Settings читает .env.local и переменные REVIEW_PRODUCER_*.
    # Некорректное значение остановит запуск понятной ValidationError.
    settings = ReviewProducerSettings()

    # Создаём один MongoClient на весь процесс.
    # MongoClient потокобезопасен и сам управляет пулом сетевых соединений.
    mongo_client = MongoClient(
        # Pydantic MongoDsn преобразуем в обычную строку для PyMongo.
        str(settings.mongo_uri),
        # Даты из BSON должны возвращаться как timezone-aware UTC datetime.
        tz_aware=True,
        # Ограничиваем время поиска доступного MongoDB-сервера.
        serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
    )

    # Пока publisher не создан, здесь None. Это позволяет безопасно выполнить
    # finally, даже если ошибка возникнет посередине создания зависимостей.
    publisher = None

    # Всё после создания MongoClient помещаем в try, чтобы client точно закрылся.
    try:
        # Передаём общий client reader-у. Reader понимает, что client чужой,
        # поэтому не будет самостоятельно закрывать его.
        reader = MongoReviewReader(settings, client=mongo_client)

        # Checkpoint store использует тот же client и тот же пул соединений.
        checkpoint_store = MongoCheckpointStore(settings, client=mongo_client)

        # Kafka Publisher сам создаёт confluent-kafka Producer из settings.
        publisher = KafkaReviewPublisher(settings)

        # Composition Root закончен: все конкретные зависимости готовы и
        # передаются координатору через его конструктор.
        review_producer = ReviewProducer(
            settings,
            reader,
            publisher,
            checkpoint_store,
        )

        # MongoClient ленивый, поэтому принудительно проверяем соединение сейчас.
        # Так ошибка адреса или пароля обнаружится при запуске, а не позже в цикле.
        reader.ping()

        # Не выводим mongo_uri: внутри него могут находиться логин и пароль.
        LOG.info(
            "starting ReviewProducer: collection=%s topic=%s checkpoint=%s",
            settings.reviews_collection,
            settings.output_topic,
            settings.checkpoint_id,
        )

        # Метод блокирует текущий поток и обрабатывает новые отзывы постоянно.
        review_producer.run()

    # В PowerShell нажатие Ctrl+C превращается в KeyboardInterrupt.
    except KeyboardInterrupt:
        # Это штатная остановка пользователем, поэтому используем INFO, не ERROR.
        LOG.info("ReviewProducer shutdown requested by user")

    # finally выполнится при Ctrl+C и при любой необработанной ошибке.
    finally:
        # Publisher мог не успеть создаться, поэтому сначала проверяем None.
        if publisher is not None:
            # В обычном пути publish_batch уже вызвал flush. Этот flush —
            # дополнительная страховка на случай остановки посередине отправки.
            remaining = publisher.producer.flush(
                settings.delivery_timeout_seconds
            )

            # Ненулевой результат означает, что часть сообщений осталась в
            # локальной очереди. Их checkpoint всё равно не был подтверждён,
            # поэтому после перезапуска они будут прочитаны и отправлены снова.
            if remaining:
                LOG.warning(
                    "Kafka producer stopped with %s undelivered message(s)",
                    remaining,
                )

        # Закрываем общий MongoClient ровно в одном месте — там, где создали.
        mongo_client.close()

        # Сообщение полезно при ручном запуске и в логах контейнера.
        LOG.info("ReviewProducer stopped")


# Python устанавливает __name__ == "__main__", когда пакет запускают через -m.
if __name__ == "__main__":
    # Благодаря этому работает команда:
    # python -m kinopoisk_classifier.review_producer
    main()
