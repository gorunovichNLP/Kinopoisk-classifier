"""Интеграционный тест checkpoint store с настоящей MongoDB."""

# os нужен для чтения переменной окружения RUN_MONGO_INTEGRATION.
import os

# unittest — стандартный тестовый фреймворк Python.
import unittest

# uuid создаёт уникальные имена, чтобы разные запуски теста не мешали друг другу.
import uuid

# ObjectId нужен, чтобы проверить реальный BSON-тип поля в MongoDB.
from bson import ObjectId

# MongoClient используется для прямой проверки MongoDB и очистки тестовых данных.
from pymongo import MongoClient

# CheckpointRegressionError должен возникнуть при движении указателя назад.
# MongoCheckpointStore — компонент, который проверяет этот тест.
from kinopoisk_classifier.review_producer.checkpoint import (
    CheckpointRegressionError,
    MongoCheckpointStore,
)

# Settings дают тесту те же настройки, которые использует рабочий сервис.
from kinopoisk_classifier.review_producer.config import ReviewProducerSettings


# Декоратор полностью пропускает класс теста без специального флага.
# Поэтому обычный запуск тестов не требует работающей MongoDB.
@unittest.skipUnless(
    # Тест запустится только при точном строковом значении "1".
    os.getenv("RUN_MONGO_INTEGRATION") == "1",
    # Эта строка объясняет причину skipped в выводе unittest.
    "set RUN_MONGO_INTEGRATION=1 to use the Docker MongoDB",
)
# Наследование от unittest.TestCase добавляет методы assertEqual и другие assert.
class MongoCheckpointStoreIntegrationTest(unittest.TestCase):
    # Один тест проверяет полный жизненный цикл checkpoint в настоящей MongoDB.
    def test_saves_loads_and_advances_checkpoint(self):
        # Уникальная коллекция не пересекается с настоящими pipeline_checkpoints.
        collection_name = f"checkpoints_integration_{uuid.uuid4().hex}"

        # Создаём настройки специально для этого запуска теста.
        settings = ReviewProducerSettings(
            # Store будет писать только во временную тестовую коллекцию.
            checkpoints_collection=collection_name,
            # Уникальный _id дополнительно изолирует checkpoint этого теста.
            checkpoint_id=f"checkpoint-integration-{uuid.uuid4().hex}",
            # Не читаем случайные значения из локального .env-файла.
            _env_file=None,
        )

        # Этот client нужен тесту для просмотра сырого BSON-документа.
        # tz_aware=True возвращает даты вместе с информацией о timezone.
        seed_client = MongoClient(str(settings.mongo_uri), tz_aware=True)

        # Store создаёт свой MongoClient и выполняет проверяемую бизнес-логику.
        store = MongoCheckpointStore(settings)

        # Два последовательных ObjectId изображают два отзыва по порядку чтения.
        first_id = "66c0f12a9d2b6e41f1701201"
        second_id = "66c0f12a9d2b6e41f1701202"

        # finally обязан очистить ресурсы, даже если один assert упадёт.
        try:
            # В новой коллекции checkpoint ещё нет: это состояние первого запуска.
            self.assertIsNone(store.load())

            # Сохраняем id первого якобы подтверждённого Kafka отзыва.
            store.save(first_id)

            # Продвигаем checkpoint ко второму отзыву и получаем Pydantic-модель.
            saved = store.save(second_id)

            # Store должен вернуть именно последний сохранённый review_id.
            self.assertEqual(saved.last_review_id, second_id)

            # Обращаемся к той же базе через отдельный проверочный client.
            database = seed_client[settings.mongo_database]

            # Выбираем уникальную коллекцию этого теста.
            collection = database[collection_name]

            # Получаем сырой MongoDB-документ без преобразования в checkpoint-модель.
            raw = collection.find_one({"_id": settings.checkpoint_id})

            # В Pydantic-модели last_review_id — строка, но в BSON он должен
            # физически храниться как настоящий MongoDB ObjectId.
            self.assertIsInstance(raw["last_review_id"], ObjectId)

            # Проверяем защиту от ошибочного движения со второго id обратно к первому.
            with self.assertRaises(CheckpointRegressionError):
                # Внутри контекстного менеджера ожидается указанное исключение.
                store.save(first_id)

        # Этот блок выполнится и при успехе, и при ошибке теста.
        finally:
            # Store сам создал свой MongoClient, поэтому просим его закрыться.
            store.close()

            # Удаляем только уникальную временную коллекцию этого теста.
            # Реальные reviews и pipeline_checkpoints остаются нетронутыми.
            seed_client[settings.mongo_database].drop_collection(collection_name)

            # Закрываем второй MongoClient, созданный непосредственно тестом.
            seed_client.close()


# Этот блок позволяет запустить файл напрямую командой python path/to/file.py.
if __name__ == "__main__":
    # unittest.main найдёт TestCase в этом модуле и выполнит его тесты.
    unittest.main()
