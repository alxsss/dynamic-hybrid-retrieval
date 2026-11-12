from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.dynamic_dense import DynamicHNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.index import ApplyStatus
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

NOW = datetime(2026, 11, 5, tzinfo=UTC)


def document(document_id: str, title: str, body: str, version: int = 1) -> Document:
    return Document(
        document_id=document_id,
        title=title,
        body=body,
        updated_at=NOW,
        version=version,
    )


def upsert(sequence: int, item: Document) -> IndexEvent:
    return IndexEvent(
        sequence_number=sequence,
        operation=Operation.UPSERT,
        document_id=item.document_id,
        version=item.version,
        event_time=NOW,
        document=item,
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
def baseline() -> HNSWIndex:
    return HNSWIndex.build(
        [
            document("doc-a", "Search retrieval", "index ranking systems"),
            document("doc-b", "Kitchen", "bread flour cooking"),
        ],
        HashingEmbedding(128),
    )


def dynamic(tmp_path: Path, baseline: HNSWIndex) -> DynamicHNSWIndex:
    return DynamicHNSWIndex(baseline, EventLog(tmp_path / "events.jsonl", fsync=False))


def test_insert_is_immediately_searchable(tmp_path: Path, baseline: HNSWIndex) -> None:
    index = dynamic(tmp_path, baseline)
    result = index.apply(upsert(1, document("doc-c", "Vector database", "semantic nearest")))

    assert result.status is ApplyStatus.APPLIED
    assert index.search("semantic nearest", exact=True)[0].document_id == "doc-c"
    assert index.statistics.delta_document_count == 1


def test_update_masks_old_baseline_vector(tmp_path: Path, baseline: HNSWIndex) -> None:
    index = dynamic(tmp_path, baseline)
    index.apply(upsert(1, document("doc-a", "Ocean wildlife", "whales dolphins", 2)))

    assert index.search("whales dolphins", exact=True)[0].document_id == "doc-a"
    old_query_results = index.search("search retrieval", exact=True)
    assert all(item.document_id != "doc-a" or item.version == 2 for item in old_query_results)
    assert sum(item.document_id == "doc-a" for item in old_query_results) <= 1
    assert index.statistics.masked_baseline_count == 1


def test_delete_hides_baseline_and_delta_vectors(tmp_path: Path, baseline: HNSWIndex) -> None:
    index = dynamic(tmp_path, baseline)
    index.apply(delete(1, "doc-a", 2))
    index.apply(upsert(2, document("doc-c", "Vector", "semantic nearest")))
    index.apply(delete(3, "doc-c", 2))

    results = index.search("search retrieval semantic nearest", exact=True)
    assert all(item.document_id not in {"doc-a", "doc-c"} for item in results)
    assert index.tombstones == {"doc-a": 2, "doc-c": 2}
    assert index.statistics.live_document_count == 1


def test_higher_version_recreates_deleted_document(tmp_path: Path, baseline: HNSWIndex) -> None:
    index = dynamic(tmp_path, baseline)
    index.apply(delete(1, "doc-a", 2))
    index.apply(upsert(2, document("doc-a", "Restored", "new vector content", 3)))

    assert index.search("restored content", exact=True)[0].document_id == "doc-a"
    assert index.tombstones == {}


def test_restart_replays_dense_delta_and_tombstones(tmp_path: Path, baseline: HNSWIndex) -> None:
    event_log = EventLog(tmp_path / "events.jsonl", fsync=False)
    original = DynamicHNSWIndex(baseline, event_log)
    original.apply(upsert(1, document("doc-c", "Semantic", "recovered vector")))
    original.apply(delete(2, "doc-a", 2))

    recovered = DynamicHNSWIndex(baseline, EventLog(event_log.path, fsync=False))

    assert recovered.statistics == original.statistics
    assert recovered.tombstones == original.tombstones
    assert recovered.search("recovered vector", exact=True) == original.search(
        "recovered vector", exact=True
    )


def test_preappended_shared_wal_event_is_still_applied(tmp_path: Path, baseline: HNSWIndex) -> None:
    event_log = EventLog(tmp_path / "events.jsonl", fsync=False)
    index = DynamicHNSWIndex(baseline, event_log)
    event = upsert(1, document("doc-c", "Semantic", "shared wal vector"))
    event_log.append(event)

    result = index.apply(event)

    assert result.status is ApplyStatus.APPLIED
    assert index.search("shared wal", exact=True)[0].document_id == "doc-c"


def test_compaction_physically_removes_deleted_and_old_versions(
    tmp_path: Path, baseline: HNSWIndex
) -> None:
    index = dynamic(tmp_path, baseline)
    index.apply(upsert(1, document("doc-a", "Updated", "fresh semantic", 2)))
    index.apply(delete(2, "doc-b", 2))
    index.apply(upsert(3, document("doc-c", "Inserted", "vector candidate")))
    queries = ("fresh semantic", "bread cooking", "vector candidate")
    expected = {query: index.search(query, exact=True) for query in queries}

    result = index.compact_to(
        tmp_path / "dense-000002",
        validation_queries=queries,
    )
    compacted = HNSWIndex.load(result.generation_path, baseline.embedding_model)

    assert result.high_water_sequence == 3
    assert result.document_count == 2
    assert {item.document_id for item in compacted.documents} == {"doc-a", "doc-c"}
    assert {query: compacted.search(query, exact=True) for query in queries} == expected


def test_stale_and_duplicate_events_do_not_change_state(
    tmp_path: Path, baseline: HNSWIndex
) -> None:
    index = dynamic(tmp_path, baseline)
    event = upsert(1, document("doc-a", "Updated", "fresh", 2))
    assert index.apply(event).status is ApplyStatus.APPLIED
    before = index.statistics
    assert index.apply(event).status is ApplyStatus.DUPLICATE
    assert index.apply(upsert(2, document("doc-a", "Old", "stale", 1))).status is ApplyStatus.STALE
    assert index.statistics == before
