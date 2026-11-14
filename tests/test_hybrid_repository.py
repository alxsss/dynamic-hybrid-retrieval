from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.hybrid_repository import (
    ACTIVE_MANIFEST_FILE,
    DENSE_DIRECTORY,
    GENERATION_MANIFEST_FILE,
    LEXICAL_DIRECTORY,
    HybridCompactionValidationError,
    HybridIndexRepository,
    HybridRepositoryCorruptionError,
    HybridRepositoryError,
    NoHybridRollbackGenerationError,
)
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)


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


@pytest.fixture
def embedding() -> HashingEmbedding:
    return HashingEmbedding(128)


@pytest.fixture
def documents() -> list[Document]:
    return [
        document("doc-a", "search retrieval ranking"),
        document("doc-b", "semantic vector similarity"),
    ]


def initialized(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> tuple[HybridIndexRepository, EventLog]:
    repository = HybridIndexRepository(tmp_path / "hybrid-repository")
    repository.initialize(
        BM25Index.build(documents),
        HNSWIndex.build(documents, embedding),
    )
    return repository, EventLog(tmp_path / "events.jsonl", fsync=False)


def test_initialize_persists_one_complete_generation(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository, event_log = initialized(tmp_path, documents, embedding)

    manifest = repository.read_manifest()
    generation_path = repository.generations_path / manifest.active_generation
    loaded = repository.load_active(event_log, embedding)

    assert manifest.active_generation == "generation-000001"
    assert manifest.active_sequence_number == 0
    assert manifest.previous_generation is None
    assert (generation_path / LEXICAL_DIRECTORY / "manifest.json").is_file()
    assert (generation_path / DENSE_DIRECTORY / "manifest.json").is_file()
    assert (generation_path / GENERATION_MANIFEST_FILE).is_file()
    assert loaded.statistics.healthy
    assert loaded.statistics.live_document_count == 2


def test_initialize_rejects_mismatched_artifacts_and_existing_repository(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository = HybridIndexRepository(tmp_path / "hybrid-repository")
    with pytest.raises(HybridRepositoryError, match="different documents"):
        repository.initialize(
            BM25Index.build(documents),
            HNSWIndex.build([document("other", "other")], embedding),
        )

    repository.initialize(
        BM25Index.build(documents),
        HNSWIndex.build(documents, embedding),
    )
    with pytest.raises(FileExistsError):
        repository.initialize(
            BM25Index.build(documents),
            HNSWIndex.build(documents, embedding),
        )


def test_compaction_activates_both_artifacts_at_one_high_water_mark(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository, event_log = initialized(tmp_path, documents, embedding)
    hybrid = repository.load_active(event_log, embedding)
    hybrid.apply(upsert(1, document("doc-a", "updated distributed search", 2)))
    hybrid.apply(upsert(2, document("doc-c", "fresh ocean wildlife")))
    hybrid.apply(delete(3, "doc-b", 2))

    result = repository.compact(
        hybrid,
        validation_queries=("distributed search", "ocean wildlife", "semantic vector"),
    )
    compacted = repository.load_active(event_log, embedding)

    assert result.manifest.active_generation == "generation-000002"
    assert result.manifest.active_sequence_number == 3
    assert result.manifest.previous_generation == "generation-000001"
    assert result.compacted_document_count == 2
    assert result.validated_query_count == 3
    assert compacted.statistics.healthy
    assert compacted.statistics.last_sequence_number == 3
    assert compacted.statistics.lexical_delta_document_count == 0
    assert compacted.statistics.dense_delta_document_count == 0
    assert compacted.statistics.tombstone_count == 0
    assert all(item.document_id != "doc-b" for item in compacted.search("semantic vector"))
    assert compacted.search("distributed search", exact_dense=True)[0].version == 2


def test_rollback_replays_events_into_both_old_baselines(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository, event_log = initialized(tmp_path, documents, embedding)
    hybrid = repository.load_active(event_log, embedding)
    hybrid.apply(upsert(1, document("doc-a", "new replacement", 2)))
    repository.compact(hybrid, validation_queries=("replacement",))

    manifest = repository.rollback()
    rolled_back = repository.load_active(event_log, embedding)

    assert manifest.active_generation == "generation-000001"
    assert manifest.active_sequence_number == 0
    assert rolled_back.statistics.healthy
    assert rolled_back.statistics.last_sequence_number == 1
    assert rolled_back.statistics.lexical_delta_document_count == 1
    assert rolled_back.statistics.dense_delta_document_count == 1
    assert rolled_back.search("replacement", exact_dense=True)[0].version == 2


def test_rollback_without_previous_generation_fails(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository, _ = initialized(tmp_path, documents, embedding)

    with pytest.raises(NoHybridRollbackGenerationError):
        repository.rollback()


def test_failed_validation_preserves_active_and_removes_candidate(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, event_log = initialized(tmp_path, documents, embedding)
    hybrid = repository.load_active(event_log, embedding)
    hybrid.apply(upsert(1, document("doc-c", "candidate")))
    original = repository.read_manifest()

    def reject(*args: object, **kwargs: object) -> None:
        raise HybridCompactionValidationError("injected validation failure")

    monkeypatch.setattr(repository, "_validate_candidate", reject)
    with pytest.raises(HybridCompactionValidationError, match="injected"):
        repository.compact(hybrid, validation_queries=("candidate",))

    assert repository.read_manifest() == original
    assert sorted(path.name for path in repository.generations_path.iterdir()) == [
        "generation-000001"
    ]


def test_late_event_replays_after_compaction_snapshot(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, event_log = initialized(tmp_path, documents, embedding)
    hybrid = repository.load_active(event_log, embedding)
    hybrid.apply(upsert(1, document("doc-c", "at snapshot")))
    original_validation = repository._validate_candidate

    def apply_late_event(*args: object, **kwargs: object) -> None:
        hybrid.apply(upsert(2, document("doc-d", "after snapshot")))
        original_validation(*args, **kwargs)

    monkeypatch.setattr(repository, "_validate_candidate", apply_late_event)
    result = repository.compact(hybrid, validation_queries=("snapshot",))
    reloaded = repository.load_active(event_log, embedding)

    assert result.manifest.active_sequence_number == 1
    assert reloaded.statistics.last_sequence_number == 2
    assert reloaded.statistics.lexical_delta_document_count == 1
    assert reloaded.statistics.dense_delta_document_count == 1
    assert reloaded.search("after snapshot", exact_dense=True)[0].document_id == "doc-d"


def test_active_and_generation_corruption_are_detected(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository, _ = initialized(tmp_path, documents, embedding)
    active_contents = repository.manifest_path.read_bytes()
    repository.manifest_path.write_bytes(
        active_contents.replace(b"generation-000001", b"generation-999999")
    )
    with pytest.raises(HybridRepositoryCorruptionError, match="checksum"):
        repository.read_manifest()

    repository.manifest_path.write_bytes(active_contents)
    generation_path = repository.generations_path / "generation-000001"
    lexical_manifest = generation_path / LEXICAL_DIRECTORY / "manifest.json"
    lexical_manifest.write_bytes(lexical_manifest.read_bytes() + b" ")
    with pytest.raises(HybridRepositoryCorruptionError, match="lexical manifest checksum"):
        repository.read_manifest()


def test_incompatible_embedding_is_rejected_before_loading(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository, event_log = initialized(tmp_path, documents, embedding)

    with pytest.raises(HybridRepositoryError, match="embedding model"):
        repository.load_active(event_log, HashingEmbedding(64))


def test_baseline_sequence_mismatch_rejects_compaction(
    tmp_path: Path,
    documents: list[Document],
    embedding: HashingEmbedding,
) -> None:
    repository, event_log = initialized(tmp_path, documents, embedding)
    hybrid = repository.load_active(event_log, embedding)
    hybrid.apply(upsert(1, document("doc-c", "advance high water")))
    repository.compact(hybrid)

    with pytest.raises(HybridRepositoryError, match="baseline sequence"):
        repository.compact(hybrid)


def test_active_manifest_file_name_is_stable() -> None:
    assert ACTIVE_MANIFEST_FILE == "active.json"
