from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.compaction import (
    ACTIVE_MANIFEST_FILE,
    CompactionValidationError,
    IndexRepository,
    NoRollbackGenerationError,
    RepositoryCorruptionError,
)
from dynamic_hybrid_retrieval.event_log import EventLog
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
            document("doc-a", "Old search", "baseline retrieval document"),
            document("doc-b", "Remove me", "obsolete baseline document"),
        ]
    )


def initialized_repository(
    tmp_path: Path,
    baseline: BM25Index,
) -> tuple[IndexRepository, EventLog]:
    repository = IndexRepository(tmp_path / "index-repository")
    repository.initialize(baseline)
    event_log = EventLog(tmp_path / "events.jsonl")
    return repository, event_log


def test_initialize_persists_first_generation_and_manifest(
    tmp_path: Path, baseline: BM25Index
) -> None:
    repository, event_log = initialized_repository(tmp_path, baseline)

    manifest = repository.read_manifest()
    loaded = repository.load_active(event_log)

    assert manifest.active_generation == "generation-000001"
    assert manifest.active_sequence_number == 0
    assert manifest.previous_generation is None
    assert (repository.generations_path / manifest.active_generation).is_dir()
    assert loaded.search("baseline retrieval") == baseline.search("baseline retrieval")


def test_initialize_refuses_existing_repository(tmp_path: Path, baseline: BM25Index) -> None:
    repository, _ = initialized_repository(tmp_path, baseline)

    with pytest.raises(FileExistsError):
        repository.initialize(baseline)


def test_compaction_materializes_update_insert_and_delete(
    tmp_path: Path, baseline: BM25Index
) -> None:
    repository, event_log = initialized_repository(tmp_path, baseline)
    dynamic = repository.load_active(event_log)
    dynamic.apply(upsert(1, document("doc-a", "Realtime search", "fresh retrieval", version=2)))
    dynamic.apply(upsert(2, document("doc-c", "New document", "delta candidate")))
    dynamic.apply(delete(3, "doc-b", 2))

    result = repository.compact(
        dynamic,
        validation_queries=("realtime retrieval", "delta candidate", "obsolete"),
    )
    compacted = repository.load_active(event_log)

    assert result.manifest.active_generation == "generation-000002"
    assert result.manifest.active_sequence_number == 3
    assert result.manifest.previous_generation == "generation-000001"
    assert result.compacted_document_count == 2
    assert result.validated_query_count == 3
    assert compacted.statistics.delta_document_count == 0
    assert compacted.statistics.tombstone_count == 0
    assert compacted.baseline.documents["doc-a"].version == 2
    assert "doc-b" not in compacted.baseline.documents
    assert compacted.baseline.documents["doc-c"].version == 1
    assert compacted.search("obsolete") == []


def test_old_generation_remains_unchanged_after_compaction(
    tmp_path: Path, baseline: BM25Index
) -> None:
    repository, event_log = initialized_repository(tmp_path, baseline)
    dynamic = repository.load_active(event_log)
    dynamic.apply(upsert(1, document("doc-a", "New", "replacement", version=2)))

    repository.compact(dynamic, validation_queries=("replacement",))
    old_baseline = BM25Index.load(repository.generations_path / "generation-000001")

    assert old_baseline.documents["doc-a"].version == 1
    assert old_baseline.search("replacement") == []


def test_rollback_reactivates_previous_generation_and_replays_events(
    tmp_path: Path, baseline: BM25Index
) -> None:
    repository, event_log = initialized_repository(tmp_path, baseline)
    dynamic = repository.load_active(event_log)
    dynamic.apply(upsert(1, document("doc-a", "New", "replacement", version=2)))
    repository.compact(dynamic, validation_queries=("replacement",))

    rolled_back_manifest = repository.rollback()
    rolled_back = repository.load_active(event_log)

    assert rolled_back_manifest.active_generation == "generation-000001"
    assert rolled_back_manifest.active_sequence_number == 0
    assert rolled_back.statistics.delta_document_count == 1
    assert [result.document_id for result in rolled_back.search("replacement")] == ["doc-a"]


def test_rollback_without_previous_generation_is_rejected(
    tmp_path: Path, baseline: BM25Index
) -> None:
    repository, _ = initialized_repository(tmp_path, baseline)

    with pytest.raises(NoRollbackGenerationError):
        repository.rollback()


def test_failed_validation_preserves_active_generation_and_removes_candidate(
    tmp_path: Path,
    baseline: BM25Index,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, event_log = initialized_repository(tmp_path, baseline)
    dynamic = repository.load_active(event_log)
    original_manifest = repository.read_manifest()

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise CompactionValidationError("injected validation failure")

    monkeypatch.setattr(repository, "_validate_compaction", fail_validation)

    with pytest.raises(CompactionValidationError, match="injected"):
        repository.compact(dynamic, validation_queries=("baseline",))

    assert repository.read_manifest() == original_manifest
    assert sorted(path.name for path in repository.generations_path.iterdir()) == [
        "generation-000001"
    ]


def test_manifest_corruption_is_detected(tmp_path: Path, baseline: BM25Index) -> None:
    repository, _ = initialized_repository(tmp_path, baseline)
    manifest_path = repository.root / ACTIVE_MANIFEST_FILE
    contents = manifest_path.read_bytes()
    manifest_path.write_bytes(contents.replace(b"generation-000001", b"generation-999999"))

    with pytest.raises(RepositoryCorruptionError, match="checksum"):
        repository.read_manifest()


def test_events_after_compaction_high_water_replay_as_new_delta(
    tmp_path: Path,
    baseline: BM25Index,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, event_log = initialized_repository(tmp_path, baseline)
    dynamic = repository.load_active(event_log)
    dynamic.apply(upsert(1, document("doc-c", "At snapshot", "compacted event")))
    original_validation = repository._validate_compaction

    def apply_late_event(*args: object, **kwargs: object) -> None:
        dynamic.apply(upsert(2, document("doc-d", "Late", "post snapshot event")))
        original_validation(*args, **kwargs)

    monkeypatch.setattr(repository, "_validate_compaction", apply_late_event)

    result = repository.compact(dynamic, validation_queries=("compacted",))
    reloaded = repository.load_active(event_log)

    assert result.manifest.active_sequence_number == 1
    assert reloaded.statistics.delta_document_count == 1
    assert [item.document_id for item in reloaded.search("post")] == ["doc-d"]


def test_compaction_preserves_ordered_scores(tmp_path: Path, baseline: BM25Index) -> None:
    repository, event_log = initialized_repository(tmp_path, baseline)
    dynamic = repository.load_active(event_log)
    dynamic.apply(upsert(1, document("doc-c", "Search", "retrieval baseline")))
    queries = ("search", "baseline retrieval", "missing")
    expected = {query: dynamic.search(query) for query in queries}

    repository.compact(dynamic, validation_queries=queries)
    compacted = repository.load_active(event_log)

    assert {query: compacted.search(query) for query in queries} == expected
