"""Pydantic-модели внешних данных inference-пайплайна.

Модели валидируют данные на границах сервисов. Kafka-контракты запрещают
неизвестные поля, а MongoReview игнорирует дополнительные поля исходного
документа, которые Producer v1 не использует.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from shared.contracts import ID2LABEL


OBJECT_ID_PATTERN = r"^[0-9a-f]{24}$"
REVIEW_EVENT_ID_PATTERN = r"^review:[0-9a-f]{24}$"
PREDICTION_ID_PATTERN = r"^[0-9a-f]{64}$"
PROBABILITY_TOLERANCE = 1e-6


def _normalize_object_id(value: object) -> str:
    """Принимает bson.ObjectId без прямой зависимости shared-кода от PyMongo."""

    normalized = str(value).lower()
    if re.fullmatch(OBJECT_ID_PATTERN, normalized) is None:
        raise ValueError("must be a 24-character Mongo ObjectId")
    return normalized


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must contain timezone information")
    return value.astimezone(timezone.utc)


ObjectIdString = Annotated[
    str,
    BeforeValidator(_normalize_object_id),
    StringConstraints(pattern=OBJECT_ID_PATTERN),
]
ReviewEventId = Annotated[
    str,
    StringConstraints(pattern=REVIEW_EVENT_ID_PATTERN),
]
PredictionId = Annotated[
    str,
    StringConstraints(pattern=PREDICTION_ID_PATTERN),
]
NonBlankString = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_non_blank),
]
UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]
Probability = Annotated[
    float,
    Field(ge=0.0, le=1.0, allow_inf_nan=False),
]
PositiveFiniteFloat = Annotated[
    float,
    Field(gt=0.0, allow_inf_nan=False),
]


def make_review_event_id(review_id: str) -> str:
    return f"review:{review_id}"


def make_prediction_id(
    source_event_id: str,
    model_name: str,
    model_version: str,
) -> str:
    """Строит стабильный id для идемпотентной записи prediction history."""

    raw = f"{source_event_id}\0{model_name}\0{model_version}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MongoReview(BaseModel):
    """Нормализованное представление неизменяемого документа `reviews`."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )

    review_id: ObjectIdString = Field(alias="_id")
    text: NonBlankString
    movie_id: NonBlankString | None = None
    created_at: UtcDatetime | None = None


class ReviewEventV1(BaseModel):
    """Value топика `kinopoisk.reviews.v1`."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    event_id: ReviewEventId
    review_id: ObjectIdString
    movie_id: NonBlankString | None = None
    text: NonBlankString
    source_created_at: UtcDatetime | None = None
    emitted_at: UtcDatetime

    @model_validator(mode="after")
    def event_id_matches_review(self) -> Self:
        expected = make_review_event_id(self.review_id)
        if self.event_id != expected:
            raise ValueError(f"event_id must be {expected!r}")
        return self


class ClassProbabilities(BaseModel):
    """Вероятности в фиксированном порядке классов neg/neu/pos."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    neg: Probability
    neu: Probability
    pos: Probability

    def for_sentiment(self, sentiment: str) -> float:
        return float(getattr(self, sentiment))

    def total(self) -> float:
        return self.neg + self.neu + self.pos


class InferenceModelVersion(BaseModel):
    """Версия модели, фактически загруженная worker-ом из Registry."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: NonBlankString
    version: NonBlankString
    run_id: NonBlankString | None = None


class LoadedModelMetadata(InferenceModelVersion):
    """Registry-метаданные модели, загруженной runtime-ом из MinIO."""

    run_id: NonBlankString
    registry_uri: NonBlankString
    artifact_store_uri: NonBlankString


class DecisionConfig(BaseModel):
    """Serving-часть `thresholds.json`; train-метрики намеренно игнорируются."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    temperature: PositiveFiniteFloat
    strategy: Literal["argmax", "class_bias", "thresholds"]
    params: dict[str, Any] = Field(default_factory=dict)


class SentimentPrediction(BaseModel):
    """Результат runtime до добавления Kafka event metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sentiment: Literal["neg", "neu", "pos"]
    label_id: Literal[0, 1, 2]
    confidence: Probability
    probabilities: ClassProbabilities

    @model_validator(mode="after")
    def result_is_consistent(self) -> Self:
        expected_sentiment = ID2LABEL[self.label_id]
        if self.sentiment != expected_sentiment:
            raise ValueError(
                f"label_id={self.label_id} requires sentiment={expected_sentiment!r}"
            )

        if abs(self.probabilities.total() - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("probabilities must sum to 1")

        selected_probability = self.probabilities.for_sentiment(self.sentiment)
        if abs(self.confidence - selected_probability) > PROBABILITY_TOLERANCE:
            raise ValueError("confidence must equal the selected class probability")

        return self


class PredictionEventV1(SentimentPrediction):
    """Value топика `kinopoisk.predictions.v1`."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    prediction_id: PredictionId
    source_event_id: ReviewEventId
    review_id: ObjectIdString
    movie_id: NonBlankString | None = None
    source_created_at: UtcDatetime | None = None
    model: InferenceModelVersion
    predicted_at: UtcDatetime

    @model_validator(mode="after")
    def values_are_consistent(self) -> Self:
        expected_event_id = make_review_event_id(self.review_id)
        if self.source_event_id != expected_event_id:
            raise ValueError(f"source_event_id must be {expected_event_id!r}")

        expected_prediction_id = make_prediction_id(
            self.source_event_id,
            self.model.name,
            self.model.version,
        )
        if self.prediction_id != expected_prediction_id:
            raise ValueError("prediction_id does not match event and model version")

        return self
