"""Внутренние MongoDB-документы Review Producer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kinopoisk_classifier.shared.schemas import (
    NonBlankString,
    ObjectIdString,
    UtcDatetime,
)


class ReviewProducerCheckpoint(BaseModel):
    """Последний review, полностью подтверждённый Kafka.

    Отсутствие документа означает первый запуск и чтение с начала коллекции.
    Наличие документа означает, что все отзывы до ``last_review_id``
    включительно уже получили Kafka delivery acknowledgement.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )

    checkpoint_id: NonBlankString = Field(alias="_id")
    schema_version: Literal[1]
    last_review_id: ObjectIdString
    source_collection: NonBlankString
    target_topic: NonBlankString
    updated_at: UtcDatetime
