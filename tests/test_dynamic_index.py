from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.event_log import DuplicateSequenceError, EventLog
from dynamic_hybrid_retrieval.index import ApplyStatus, DynamicIndex
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def document(
    document_id: str,
    title: str,
    body: str,
    *,
    version: int = 1,
) -> Document:
    return Document(
        document_id=document_id,
        title=title,
        body=body,
        updated_at=NOW,
        version=version,
    )


def upsert(sequence: int, doc: Document) -> IndexEvent:
    return IndexEvent(
        sequence_number=sequence,
        operation=Operation.UPSERT,
        document_id=doc.document_id,
        version=doc.version,
        event_time=NOW,
        document=doc,
    )


def delete(sequence: int, document_id: str, version: int) -> IndexEvent:
    return IndexEvent(
        sequence_number=sequence,
        operation=Operation.DELETE,
        document_id=document_id,
        version=version,
        event_time=NOW,
    )


@pytest.fixture
def baseline() -> BM25Index:
    return BM25Index.build(
        [
            document("doc-a", "Baseline search", "stable retrieval document"),
            document("doc-b", "Index architecture", "immutable baseline generation"),
        ]
    )


@pytest.fixture
def dynamic_index(baseline: BM25Index, tmp_path: Path) -> DynamicIndex:
    return DynamicIndex(baseline, EventLog(tmp_path / "events.jsonl"))


def test_baseline_documents_are_searchable(dynamic_index: DynamicIndex) -> None:
    assert [result.document_id for result in dynamic_index.search("retrieval")] == ["doc-a"]


def test_insert_is_immediately_searchable(dynamic_index: DynamicIndex) -> None:
    inserted = document("doc-c", "Fresh content", "streaming update", version=1)

    result = dynamic_index.apply(upsert(1, inserted))

    assert result.status is ApplyStatus.APPLIED
    assert [item.document_id for item in dynamic_index.search("streaming")] == ["doc-c"]
    assert dynamic_index.statistics.delta_document_count == 1
    assert dynamic_index.statistics.live_document_count == 3


def test_update_masks_old_baseline_terms_and_exposes_new_terms(
    dynamic_index: DynamicIndex,
) -> None:
    updated = document("doc-a", "Realtime search", "fresh candidate generation", version=2)

    dynamic_index.apply(upsert(1, updated))

    assert dynamic_index.search("stable retrieval") == []
    results = dynamic_index.search("realtime fresh")
    assert [result.document_id for result in results] == ["doc-a"]
    assert results[0].version == 2


def test_delete_hides_baseline_document(dynamic_index: DynamicIndex) -> None:
    result = dynamic_index.apply(delete(1, "doc-a", 2))

    assert result.status is ApplyStatus.APPLIED
    assert dynamic_index.search("stable retrieval") == []
    assert dynamic_index.tombstones == {"doc-a": 2}
    assert dynamic_index.statistics.live_document_count == 1


def test_delete_hides_delta_document(dynamic_index: DynamicIndex) -> None:
    inserted = document("doc-c", "Fresh content", "streaming update")
    dynamic_index.apply(upsert(1, inserted))

    dynamic_index.apply(delete(2, "doc-c", 2))

    assert dynamic_index.search("streaming") == []
    assert dynamic_index.statistics.delta_document_count == 0
    assert dynamic_index.statistics.tombstone_count == 1


def test_higher_version_can_recreate_deleted_document(dynamic_index: DynamicIndex) -> None:
    dynamic_index.apply(delete(1, "doc-a", 2))
    recreated = document("doc-a", "Restored", "new searchable contents", version=3)

    dynamic_index.apply(upsert(2, recreated))

    assert [result.document_id for result in dynamic_index.search("restored")] == ["doc-a"]
    assert dynamic_index.tombstones == {}


def test_stale_version_is_ignored_and_not_persisted(dynamic_index: DynamicIndex) -> None:
    stale = document("doc-a", "Stale", "should not appear", version=1)

    result = dynamic_index.apply(upsert(1, stale))

    assert result.status is ApplyStatus.STALE
    assert dynamic_index.search("should appear") == []
    assert dynamic_index.event_log.scan().events == ()


def test_exact_duplicate_event_is_idempotent(dynamic_index: DynamicIndex) -> None:
    event = upsert(1, document("doc-c", "Fresh", "new document"))

    first = dynamic_index.apply(event)
    second = dynamic_index.apply(event)

    assert first.status is ApplyStatus.APPLIED
    assert second.status is ApplyStatus.DUPLICATE
    assert len(dynamic_index.event_log.scan().events) == 1
    assert dynamic_index.statistics.live_document_count == 3


def test_conflicting_duplicate_sequence_is_rejected(dynamic_index: DynamicIndex) -> None:
    dynamic_index.apply(upsert(1, document("doc-c", "Fresh", "new document")))

    with pytest.raises(DuplicateSequenceError):
        dynamic_index.apply(upsert(1, document("doc-d", "Other", "different document")))


def test_restart_replays_event_log_to_identical_results(
    baseline: BM25Index, tmp_path: Path
) -> None:
    path = tmp_path / "events.jsonl"
    original = DynamicIndex(baseline, EventLog(path))
    original.apply(upsert(1, document("doc-a", "Updated", "recovered delta term", version=2)))
    original.apply(upsert(2, document("doc-c", "Inserted", "recovered new term")))
    original.apply(delete(3, "doc-b", 2))
    expected = original.search("recovered term")

    recovered = DynamicIndex(baseline, EventLog(path))

    assert recovered.search("recovered term") == expected
    assert recovered.statistics == original.statistics
    assert recovered.latest_versions == original.latest_versions
    assert recovered.tombstones == original.tombstones


def test_global_statistics_change_on_update_and_delete(dynamic_index: DynamicIndex) -> None:
    initial = dynamic_index.statistics
    replacement = document("doc-a", "Tiny", "replacement", version=2)

    dynamic_index.apply(upsert(1, replacement))
    after_update = dynamic_index.statistics
    dynamic_index.apply(delete(2, "doc-b", 2))
    after_delete = dynamic_index.statistics

    assert after_update.live_document_count == initial.live_document_count
    assert after_update.total_document_length < initial.total_document_length
    assert after_delete.live_document_count == 1
    assert after_delete.total_document_length == 2


def test_baseline_and_delta_share_one_idf(dynamic_index: DynamicIndex) -> None:
    dynamic_index.apply(upsert(1, document("doc-c", "Retrieval", "delta candidate generation")))

    results = dynamic_index.search("retrieval")

    assert {result.document_id for result in results} == {"doc-a", "doc-c"}
    assert results[0].score != 0


def test_query_limit_and_empty_query(dynamic_index: DynamicIndex) -> None:
    with pytest.raises(ValueError, match="at least one"):
        dynamic_index.search("search", limit=0)
    assert dynamic_index.search("---") == []


def test_empty_baseline_supports_delta_and_delete(tmp_path: Path) -> None:
    index = DynamicIndex(BM25Index.build([]), EventLog(tmp_path / "events.jsonl"))
    inserted = document("doc-a", "First", "only delta document")

    index.apply(upsert(1, inserted))
    assert [result.document_id for result in index.search("delta")] == ["doc-a"]

    index.apply(delete(2, "doc-a", 2))
    assert index.search("delta") == []
    assert index.statistics.live_document_count == 0


def test_baseline_sequence_number_skips_already_compacted_events(
    baseline: BM25Index, tmp_path: Path
) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.append(upsert(1, document("doc-c", "Old", "already compacted")))
    log.append(upsert(2, document("doc-d", "New", "must replay")))

    recovered = DynamicIndex(baseline, EventLog(path), baseline_sequence_number=1)

    assert recovered.search("already compacted") == []
    assert [result.document_id for result in recovered.search("must replay")] == ["doc-d"]


def test_negative_baseline_sequence_is_rejected(baseline: BM25Index, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        DynamicIndex(
            baseline,
            EventLog(tmp_path / "events.jsonl"),
            baseline_sequence_number=-1,
        )
