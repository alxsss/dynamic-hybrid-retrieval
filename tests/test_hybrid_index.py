from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.dynamic_dense import DynamicHNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.fusion import HybridConsistencyError
from dynamic_hybrid_retrieval.hybrid_index import (
    DynamicHybridIndex,
    HybridIndexUnhealthyError,
)
from dynamic_hybrid_retrieval.index import ApplyStatus, DynamicIndex
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

NOW = datetime(2026, 11, 13, tzinfo=UTC)


def document(document_id: str, text: str, version: int = 1) -> Document:
    return Document(
        document_id=document_id,
        title=text,
        body=text,
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


def build_hybrid(tmp_path: Path) -> DynamicHybridIndex:
    documents = [
        document("doc-a", "search retrieval ranking"),
        document("doc-b", "semantic vector similarity"),
    ]
    event_log = EventLog(tmp_path / "events.jsonl", fsync=False)
    return DynamicHybridIndex(
        DynamicIndex(BM25Index.build(documents), event_log),
        DynamicHNSWIndex(
            HNSWIndex.build(documents, HashingEmbedding(128)),
            event_log,
        ),
    )


def test_coordinated_insert_update_delete_and_snapshot(tmp_path: Path) -> None:
    hybrid = build_hybrid(tmp_path)

    assert hybrid.apply(upsert(1, document("doc-c", "fresh ocean wildlife"))).status is (
        ApplyStatus.APPLIED
    )
    assert (
        hybrid.apply(upsert(2, document("doc-a", "updated distributed search", 2))).status
        is ApplyStatus.APPLIED
    )
    assert hybrid.apply(delete(3, "doc-b", 2)).status is ApplyStatus.APPLIED

    assert hybrid.statistics.healthy
    assert hybrid.statistics.last_sequence_number == 3
    assert hybrid.statistics.live_document_count == 2
    assert all(result.document_id != "doc-b" for result in hybrid.search("semantic vector"))
    assert hybrid.search("distributed search", exact_dense=True)[0].version == 2
    snapshot = hybrid.snapshot_for_compaction(("distributed search",))
    assert snapshot.high_water_sequence == 3
    assert snapshot.lexical.documents == snapshot.dense.documents


def test_restart_recovers_both_views_from_shared_wal(tmp_path: Path) -> None:
    hybrid = build_hybrid(tmp_path)
    hybrid.apply(upsert(1, document("doc-c", "durably recoverable")))
    hybrid.apply(delete(2, "doc-a", 2))

    recovered = DynamicHybridIndex(
        DynamicIndex(hybrid.lexical.baseline, EventLog(hybrid.lexical.event_log.path, fsync=False)),
        DynamicHNSWIndex(
            hybrid.dense.baseline,
            EventLog(hybrid.dense.event_log.path, fsync=False),
        ),
    )

    assert recovered.statistics.healthy
    assert recovered.statistics.last_sequence_number == 2
    assert recovered.search("durably recoverable", exact_dense=True)[0].document_id == "doc-c"
    assert all(result.document_id != "doc-a" for result in recovered.search("search retrieval"))


def test_duplicate_and_stale_outcomes_match_on_both_views(tmp_path: Path) -> None:
    hybrid = build_hybrid(tmp_path)
    event = upsert(1, document("doc-a", "new version", 2))

    assert hybrid.apply(event).status is ApplyStatus.APPLIED
    assert hybrid.apply(event).status is ApplyStatus.DUPLICATE
    assert hybrid.apply(upsert(2, document("doc-a", "old version", 1))).status is (
        ApplyStatus.STALE
    )
    assert hybrid.statistics.healthy


def test_constructor_rejects_different_live_documents(tmp_path: Path) -> None:
    event_log = EventLog(tmp_path / "events.jsonl", fsync=False)
    lexical = DynamicIndex(BM25Index.build([document("doc-a", "lexical")]), event_log)
    dense = DynamicHNSWIndex(
        HNSWIndex.build([document("doc-b", "dense")], HashingEmbedding(64)),
        event_log,
    )

    with pytest.raises(HybridConsistencyError, match="different live documents"):
        DynamicHybridIndex(lexical, dense)


def test_constructor_requires_one_wal_and_high_water_mark(tmp_path: Path) -> None:
    item = document("doc-a", "same")
    lexical = DynamicIndex(
        BM25Index.build([item]),
        EventLog(tmp_path / "lexical.jsonl", fsync=False),
    )
    dense = DynamicHNSWIndex(
        HNSWIndex.build([item], HashingEmbedding(64)),
        EventLog(tmp_path / "dense.jsonl", fsync=False),
    )
    with pytest.raises(HybridConsistencyError, match="share one WAL"):
        DynamicHybridIndex(lexical, dense)

    shared_log = EventLog(tmp_path / "shared.jsonl", fsync=False)
    with pytest.raises(HybridConsistencyError, match="high-water"):
        DynamicHybridIndex(
            DynamicIndex(BM25Index.build([item]), shared_log, baseline_sequence_number=1),
            DynamicHNSWIndex(
                HNSWIndex.build([item], HashingEmbedding(64)),
                shared_log,
                baseline_sequence_number=2,
            ),
        )


def test_partial_apply_poisons_process_but_wal_allows_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hybrid = build_hybrid(tmp_path)
    event = upsert(1, document("doc-c", "recover after failure"))

    def fail_dense_apply(_: IndexEvent) -> None:
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr(hybrid.dense, "apply", fail_dense_apply)
    with pytest.raises(RuntimeError, match="embedding service"):
        hybrid.apply(event)

    assert not hybrid.statistics.healthy
    with pytest.raises(HybridIndexUnhealthyError, match="rebuild"):
        hybrid.search("recover")
    assert hybrid.lexical.event_log.last_sequence_number == 1

    recovered = DynamicHybridIndex(
        DynamicIndex(
            hybrid.lexical.baseline,
            EventLog(hybrid.lexical.event_log.path, fsync=False),
        ),
        DynamicHNSWIndex(
            hybrid.dense.baseline,
            EventLog(hybrid.dense.event_log.path, fsync=False),
        ),
    )
    assert recovered.statistics.healthy
    assert recovered.search("recover after failure", exact_dense=True)[0].document_id == "doc-c"


def test_direct_child_mutation_is_detected_before_search(tmp_path: Path) -> None:
    hybrid = build_hybrid(tmp_path)
    hybrid.lexical.apply(upsert(1, document("doc-c", "bypassed coordinator")))

    with pytest.raises(HybridConsistencyError, match="sequence"):
        hybrid.search("bypassed")
