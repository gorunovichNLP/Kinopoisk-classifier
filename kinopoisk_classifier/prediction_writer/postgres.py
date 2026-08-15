"""Сохранение PredictionEventV1 в PostgreSQL.

Паттерны проектирования:
- Repository: остальной код вызывает ping(), save() и close(), не зная SQL.

Архитектурные приёмы:
- Dependency Injection: готовое соединение можно передать через конструктор.
- Idempotent Write: повторный prediction_id не создаёт вторую строку.
"""

# psycopg — официальный современный PostgreSQL-драйвер для Python.
import psycopg

# Настройки содержат DSN и timeout подключения.
from kinopoisk_classifier.prediction_writer.config import PredictionWriterSettings

# Это уже проверенное Pydantic-событие из Kafka.
from kinopoisk_classifier.shared.schemas import PredictionEventV1


# SQL держим отдельно от метода, чтобы Python-логика save() читалась проще.
# Значения не вставляются в строку вручную: %s placeholders безопасно заполняет
# сам psycopg, правильно преобразуя строки, числа, None и datetime.
INSERT_PREDICTION_SQL = """
INSERT INTO sentiment.prediction_history (
    prediction_id,
    schema_version,
    source_event_id,
    review_id,
    movie_id,
    source_created_at,
    sentiment,
    label_id,
    confidence,
    probability_neg,
    probability_neu,
    probability_pos,
    model_name,
    model_version,
    model_run_id,
    predicted_at
)
VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s
)
ON CONFLICT (prediction_id) DO NOTHING
"""


class PostgresPredictionRepository:
    """Записывает валидные predictions в таблицу истории."""

    def __init__(
        self,
        settings: PredictionWriterSettings,
        *,
        connection=None,
    ) -> None:
        self.settings = settings

        # Запоминаем, кто создал соединение. Переданное снаружи соединение
        # может использовать другой компонент, поэтому закрывать его нельзя.
        self._owns_connection = connection is None

        if connection is None:
            # psycopg.connect открывает реальное сетевое соединение сразу.
            self.connection = psycopg.connect(
                str(settings.postgres_dsn),
                connect_timeout=settings.postgres_connect_timeout_seconds,
            )
        else:
            self.connection = connection

    def ping(self) -> None:
        """Выполняет простой запрос и проверяет соединение с PostgreSQL."""

        try:
            # Cursor выполняет SQL внутри транзакции текущего соединения.
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")

            # Даже SELECT начинает транзакцию. Завершаем её сразу, чтобы
            # соединение не оставалось в состоянии idle in transaction.
            self.connection.commit()
        except Exception:
            # После ошибки PostgreSQL не принимает новые команды до rollback.
            self.connection.rollback()
            raise

    def save(self, prediction: PredictionEventV1) -> bool:
        """Сохраняет prediction и возвращает ``True``, если строка новая."""

        # Порядок значений точно соответствует порядку колонок в SQL выше.
        values = (
            prediction.prediction_id,
            prediction.schema_version,
            prediction.source_event_id,
            prediction.review_id,
            prediction.movie_id,
            prediction.source_created_at,
            prediction.sentiment,
            prediction.label_id,
            prediction.confidence,
            prediction.probabilities.neg,
            prediction.probabilities.neu,
            prediction.probabilities.pos,
            prediction.model.name,
            prediction.model.version,
            prediction.model.run_id,
            prediction.predicted_at,
        )

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(INSERT_PREDICTION_SQL, values)

                # rowcount=1: PostgreSQL добавил строку.
                # rowcount=0: ON CONFLICT увидел существующий prediction_id.
                inserted = cursor.rowcount == 1

            # Только после commit запись становится видна другим соединениям.
            self.connection.commit()
            return inserted
        except Exception:
            # Отменяем незавершённую транзакцию и сохраняем исходную ошибку.
            self.connection.rollback()
            raise

    def close(self) -> None:
        """Закрывает только соединение, созданное самим repository."""

        if self._owns_connection:
            self.connection.close()
