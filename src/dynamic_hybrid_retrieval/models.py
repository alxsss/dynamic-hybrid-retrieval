"""Validated domain models for documents and index change events."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

import orjson
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Operation(StrEnum):
    """Mutation types accepted by the index."""

    UPSERT = "upsert"
    DELETE = "delete"


class Document(BaseModel):
    """A versioned searchable document.

    Models are frozen so an accepted event cannot change after its checksum is
    calculated or after it has been appended to the event log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    document_id: NonEmptyString
    title: str
    body: str
    updated_at: datetime
    version: Annotated[int, Field(ge=1)]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_document(self) -> Document:
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must include a timezone")
        try:
            orjson.dumps(self.metadata, option=orjson.OPT_SORT_KEYS)
        except TypeError as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        return self


class IndexEvent(BaseModel):
    """A totally ordered, immutable change to one document."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sequence_number: Annotated[int, Field(ge=1)]
    operation: Operation
    document_id: NonEmptyString
    version: Annotated[int, Field(ge=1)]
    event_time: datetime
    document: Document | None = None

    @model_validator(mode="after")
    def validate_operation_payload(self) -> IndexEvent:
        if self.event_time.tzinfo is None or self.event_time.utcoffset() is None:
            raise ValueError("event_time must include a timezone")

        if self.operation is Operation.UPSERT:
            if self.document is None:
                raise ValueError("upsert events require a document")
            if self.document.document_id != self.document_id:
                raise ValueError("event and document IDs must match")
            if self.document.version != self.version:
                raise ValueError("event and document versions must match")
        elif self.document is not None:
            raise ValueError("delete events cannot contain a document")

        return self
