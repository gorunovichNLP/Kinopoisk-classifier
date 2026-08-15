"""Исполняемая точка входа Kafka Prediction Writer.

Архитектурные паттерны:
- Composition Root: здесь создаются и связываются реальные зависимости.
- Dependency Injection: repository передаётся в PredictionWriter.
- Resource Ownership: main закрывает ресурсы, которые сам создал.

GoF Factory Method здесь не используется: main() напрямую создаёт объекты.
"""

# logging выводит понятные сообщения о запуске и остановке сервиса.
import logging

# Event позволяет интеграционному тесту штатно остановить main().
from threading import Event

# Settings читает переменные окружения PREDICTION_WRITER_*.
from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings

# Repository открывает соединение и записывает predictions в PostgreSQL.
from kinopoisk_classifier.prediction_writer.postgres import (
    PostgresPredictionRepository,
)

# PredictionWriter читает Kafka и управляет порядком commit-ов.
from kinopoisk_classifier.prediction_writer.writer import PredictionWriter


# Logger получает имя текущего Python-модуля.
LOG = logging.getLogger(__name__)


def main(stop_event: Event | None = None) -> None:
    """Создаёт зависимости, запускает Writer и освобождает ресурсы."""

    # Настраиваем единый формат логов всего процесса.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Pydantic читает environment/.env.local и сразу проверяет значения.
    settings = PredictionWriterSettings()

    # Пока объекты не созданы, храним None. Благодаря этому finally безопасен,
    # даже если подключение к PostgreSQL или создание Consumer завершится ошибкой.
    repository = None
    writer = None

    try:
        # Repository сразу открывает настоящее соединение с PostgreSQL.
        repository = PostgresPredictionRepository(settings)

        # Проверяем базу при старте, чтобы ошибка DSN/пароля обнаружилась сразу.
        repository.ping()

        # Writer создаёт Kafka Consumer и подписывается на input_topic.
        writer = PredictionWriter(settings, repository)

        # DSN не логируем: в нём могут находиться логин и пароль.
        LOG.info(
            "starting PredictionWriter: topic=%s group=%s",
            settings.input_topic,
            settings.kafka_group_id,
        )

        # Обычный запуск передаёт None и работает до Ctrl+C.
        # Интеграционный тест передаёт Event и останавливает цикл через set().
        writer.run(stop_event)

    # При ручном запуске PowerShell превращает Ctrl+C в KeyboardInterrupt.
    except KeyboardInterrupt:
        LOG.info("PredictionWriter shutdown requested by user")

    # finally выполняется и при штатной остановке, и при необработанной ошибке.
    finally:
        # Сначала закрываем Consumer: он перестаёт участвовать в consumer group.
        if writer is not None:
            writer.close()

        # Затем закрываем PostgreSQL-соединение.
        if repository is not None:
            repository.close()

        LOG.info("PredictionWriter stopped")


# Этот блок выполняется только при запуске пакета через python -m.
if __name__ == "__main__":
    # Команда: python -m kinopoisk_classifier.prediction_writer
    main()
