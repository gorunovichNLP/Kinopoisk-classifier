"""Internal schemas used by the review producer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kinopoisk_classifier.shared.schemas import (
    NonBlankString,
    ObjectIdString,
    UtcDatetime,
)


class ReviewProducerCheckpoint(BaseModel):
    """Validated persistent cursor for the review producer."""

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
