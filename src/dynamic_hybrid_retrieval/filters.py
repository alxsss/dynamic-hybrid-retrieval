"""Typed, composable metadata filters shared by all retrievers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dynamic_hybrid_retrieval.models import Document

type MetadataScalar = str | int | float | bool | None
_MISSING = object()


@runtime_checkable
class DocumentFilter(Protocol):
    def matches(self, document: Document) -> bool: ...


def _validate_path(path: str) -> tuple[str, ...]:
    segments = tuple(segment.strip() for segment in path.split("."))
    if not segments or any(not segment for segment in segments):
        raise ValueError("metadata path must contain non-empty dot-separated segments")
    return segments


def metadata_value(document: Document, path: str) -> Any:
    """Resolve a dot-separated path from document metadata."""

    current: Any = document.metadata
    for segment in _validate_path(path):
        if not isinstance(current, Mapping) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


@dataclass(frozen=True, slots=True)
class Equals:
    path: str
    value: MetadataScalar

    def __post_init__(self) -> None:
        _validate_path(self.path)

    def matches(self, document: Document) -> bool:
        actual = metadata_value(document, self.path)
        return actual is not _MISSING and type(actual) is type(self.value) and actual == self.value


@dataclass(frozen=True, slots=True)
class InValues:
    path: str
    values: tuple[MetadataScalar, ...]

    def __post_init__(self) -> None:
        _validate_path(self.path)
        if not self.values:
            raise ValueError("values cannot be empty")

    def matches(self, document: Document) -> bool:
        actual = metadata_value(document, self.path)
        return actual is not _MISSING and any(
            type(actual) is type(value) and actual == value for value in self.values
        )


@dataclass(frozen=True, slots=True)
class NumericRange:
    path: str
    minimum: int | float | None = None
    maximum: int | float | None = None
    include_minimum: bool = True
    include_maximum: bool = True

    def __post_init__(self) -> None:
        _validate_path(self.path)
        if self.minimum is None and self.maximum is None:
            raise ValueError("numeric range requires at least one bound")
        if isinstance(self.minimum, bool) or isinstance(self.maximum, bool):
            raise ValueError("boolean values are not numeric range bounds")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")

    def matches(self, document: Document) -> bool:
        actual = metadata_value(document, self.path)
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        if self.minimum is not None and (
            actual < self.minimum or (actual == self.minimum and not self.include_minimum)
        ):
            return False
        return not (
            self.maximum is not None
            and (actual > self.maximum or (actual == self.maximum and not self.include_maximum))
        )


@dataclass(frozen=True, slots=True)
class Exists:
    path: str

    def __post_init__(self) -> None:
        _validate_path(self.path)

    def matches(self, document: Document) -> bool:
        return metadata_value(document, self.path) is not _MISSING


@dataclass(frozen=True, slots=True)
class AllOf:
    filters: tuple[DocumentFilter, ...]

    def __post_init__(self) -> None:
        if not self.filters:
            raise ValueError("AllOf requires at least one filter")

    def matches(self, document: Document) -> bool:
        return all(item.matches(document) for item in self.filters)


@dataclass(frozen=True, slots=True)
class AnyOf:
    filters: tuple[DocumentFilter, ...]

    def __post_init__(self) -> None:
        if not self.filters:
            raise ValueError("AnyOf requires at least one filter")

    def matches(self, document: Document) -> bool:
        return any(item.matches(document) for item in self.filters)


@dataclass(frozen=True, slots=True)
class Not:
    filter: DocumentFilter

    def matches(self, document: Document) -> bool:
        return not self.filter.matches(document)
