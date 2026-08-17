"""Validated data contracts for the inference pipeline."""

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

from kinopoisk_classifier.shared.contracts import ID2LABEL


OBJECT_ID_PATTERN = r"^[0-9a-f]{24}$"
REVIEW_EVENT_ID_PATTERN = r"^review:[0-9a-f]{24}$"
PREDICTION_ID_PATTERN = r"^[0-9a-f]{64}$"
PROBABILITY_TOLERANCE = 1e-6


def _normalize_object_id(value: object) -> str:
    """Normalize a MongoDB ObjectId as lowercase text."""

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
NonNegativeInt = Annotated[int, Field(ge=0)]


def make_review_event_id(review_id: str) -> str:
    return f"review:{review_id}"


def make_prediction_id(
    source_event_id: str,
    model_name: str,
    model_version: str,
) -> str:
    """Build a stable identifier for an idempotent prediction write."""

    raw = f"{source_event_id}\0{model_name}\0{model_version}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MongoReview(BaseModel):
    """Normalized immutable review document."""

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
    """Versioned review event published to Kafka."""

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
    """Probabilities in canonical neg, neu, pos order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    neg: Probability
    neu: Probability
    pos: Probability

    def for_sentiment(self, sentiment: str) -> float:
        return float(getattr(self, sentiment))

    def total(self) -> float:
        return self.neg + self.neu + self.pos


class InferenceModelVersion(BaseModel):
    """Identity of the model used for inference."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: NonBlankString
    version: NonBlankString
    run_id: NonBlankString | None = None


class LoadedModelMetadata(InferenceModelVersion):
    """Registry metadata for the loaded model version."""

    run_id: NonBlankString
    registry_uri: NonBlankString
    artifact_store_uri: NonBlankString


class DecisionConfig(BaseModel):
    """Validated serving decision configuration."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    temperature: PositiveFiniteFloat
    strategy: Literal["argmax", "class_bias", "thresholds"]
    params: dict[str, Any] = Field(default_factory=dict)


class SentimentPrediction(BaseModel):
    """Validated model prediction before event metadata is added."""

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
    """Versioned prediction event published to Kafka."""

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


class DeadLetterEventV1(BaseModel):
    """Invalid Kafka input retained for investigation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    worker: Literal["sentiment-inference"]
    source_topic: NonBlankString
    source_partition: NonNegativeInt
    source_offset: NonNegativeInt
    key_base64: str | None = None
    payload_base64: str
    error_type: NonBlankString
    error_message: NonBlankString
    failed_at: UtcDatetime
