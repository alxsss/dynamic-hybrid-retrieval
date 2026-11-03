"""Deterministic synthetic documents, mutations, and search workloads."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta

from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

VOCABULARY = (
    "architecture",
    "baseline",
    "compaction",
    "consistency",
    "delta",
    "distributed",
    "document",
    "durability",
    "freshness",
    "indexing",
    "latency",
    "observability",
    "ranking",
    "recovery",
    "retrieval",
    "search",
    "storage",
    "throughput",
    "update",
    "validation",
)
DEFAULT_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _words(random_source: random.Random, count: int) -> str:
    return " ".join(random_source.choices(VOCABULARY, k=count))


def generate_documents(
    count: int,
    *,
    seed: int = 42,
    timestamp: datetime = DEFAULT_TIME,
) -> list[Document]:
    """Create a repeatable baseline corpus with stable IDs and content."""

    if count < 0:
        raise ValueError("count cannot be negative")
    random_source = random.Random(seed)
    return [
        Document(
            document_id=f"doc-{position:07d}",
            title=_words(random_source, 4),
            body=_words(random_source, 36),
            updated_at=timestamp,
            version=1,
            metadata={"synthetic": True, "ordinal": position},
        )
        for position in range(count)
    ]


def generate_events(
    baseline_documents: Sequence[Document],
    count: int,
    *,
    seed: int = 43,
    start_sequence: int = 1,
    timestamp: datetime = DEFAULT_TIME,
    delete_every: int = 10,
) -> Iterator[IndexEvent]:
    """Yield deterministic updates, inserts, and deletes.

    Every non-delete contains a unique ``eventtokenNNNNNNN`` term. That makes
    end-to-end freshness measurable without relying on internal index state.
    Deletes target distinct baseline documents and comprise 10% by default.
    Remaining events alternate between baseline updates and new inserts.
    """

    if count < 0:
        raise ValueError("count cannot be negative")
    if start_sequence < 1:
        raise ValueError("start_sequence must be positive")
    if delete_every < 2:
        raise ValueError("delete_every must be at least two")
    if count and not baseline_documents:
        raise ValueError("a non-empty baseline is required to generate events")

    random_source = random.Random(seed)
    baseline_count = len(baseline_documents)
    latest_versions = {document.document_id: document.version for document in baseline_documents}
    for offset in range(count):
        sequence = start_sequence + offset
        event_time = timestamp + timedelta(microseconds=offset)
        baseline_position = (offset // 2) % baseline_count
        baseline_document = baseline_documents[baseline_position]

        if (offset + 1) % delete_every == 0:
            delete_position = (offset // delete_every) % baseline_count
            target = baseline_documents[delete_position]
            version = latest_versions[target.document_id] + 1
            latest_versions[target.document_id] = version
            yield IndexEvent(
                sequence_number=sequence,
                operation=Operation.DELETE,
                document_id=target.document_id,
                version=version,
                event_time=event_time,
            )
            continue

        marker = event_marker(sequence)
        if offset % 2 == 0:
            document_id = baseline_document.document_id
            version = latest_versions[document_id] + 1
        else:
            document_id = f"insert-{sequence:07d}"
            version = 1
        latest_versions[document_id] = version
        updated_document = Document(
            document_id=document_id,
            title=f"{marker} {_words(random_source, 3)}",
            body=_words(random_source, 36),
            updated_at=event_time,
            version=version,
            metadata={"synthetic": True, "event_sequence": sequence},
        )
        yield IndexEvent(
            sequence_number=sequence,
            operation=Operation.UPSERT,
            document_id=document_id,
            version=version,
            event_time=event_time,
            document=updated_document,
        )


def event_marker(sequence_number: int) -> str:
    """Return the unique searchable token embedded in an upsert event."""

    return f"eventtoken{sequence_number:07d}"


def generate_queries(count: int, *, seed: int = 44) -> tuple[str, ...]:
    """Create deterministic one- and two-term queries from the shared vocabulary."""

    if count < 1:
        raise ValueError("count must be positive")
    random_source = random.Random(seed)
    return tuple(
        " ".join(random_source.sample(VOCABULARY, k=1 + position % 2)) for position in range(count)
    )
