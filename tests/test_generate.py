from __future__ import annotations

import pytest

from dynamic_hybrid_retrieval.generate import (
    event_marker,
    generate_documents,
    generate_events,
    generate_queries,
)
from dynamic_hybrid_retrieval.models import Operation


def test_documents_are_deterministic_and_unique() -> None:
    first = generate_documents(5, seed=9)
    second = generate_documents(5, seed=9)

    assert first == second
    assert len({document.document_id for document in first}) == 5


def test_events_include_updates_inserts_deletes_and_markers() -> None:
    documents = generate_documents(10)
    events = list(generate_events(documents, 10))

    assert [event.sequence_number for event in events] == list(range(1, 11))
    assert events[-1].operation is Operation.DELETE
    assert any(event.document_id.startswith("insert-") for event in events)
    for event in events:
        if event.operation is Operation.UPSERT:
            assert event.document is not None
            assert event_marker(event.sequence_number) in event.document.title


def test_generators_validate_counts() -> None:
    with pytest.raises(ValueError):
        generate_documents(-1)
    with pytest.raises(ValueError):
        list(generate_events([], 1))
    with pytest.raises(ValueError):
        generate_queries(0)


def test_long_event_stream_always_advances_document_versions() -> None:
    documents = generate_documents(2)
    events = list(generate_events(documents, 20))
    versions: dict[str, int] = {document.document_id: document.version for document in documents}

    for event in events:
        assert event.version > versions.get(event.document_id, 0)
        versions[event.document_id] = event.version
