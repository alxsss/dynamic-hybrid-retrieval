from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.compaction import IndexRepository
from dynamic_hybrid_retrieval.diagnostics import RuntimeEnvironment, inspect_repository
from dynamic_hybrid_retrieval.event_log import EventLog, encode_record
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

NOW = datetime(2026, 11, 3, tzinfo=UTC)


def document(document_id: str, version: int = 1) -> Document:
    return Document(
        document_id=document_id,
        title="production search",
        body="health recovery diagnostics",
        updated_at=NOW,
        version=version,
    )


def create_repository(tmp_path: Path) -> tuple[IndexRepository, EventLog]:
    repository = IndexRepository(tmp_path / "repository")
    repository.initialize(BM25Index.build([document("doc-1")]))
    event_log = EventLog(tmp_path / "events.jsonl", fsync=False)
    event_log.append(
        IndexEvent(
            sequence_number=1,
            operation=Operation.UPSERT,
            document_id="doc-2",
            version=1,
            event_time=NOW,
            document=document("doc-2"),
        )
    )
    return repository, event_log


def test_status_validates_and_replays_repository(tmp_path: Path) -> None:
    repository, event_log = create_repository(tmp_path)

    status = inspect_repository(repository.root, event_log.path)

    assert status.healthy
    assert status.active_generation == "generation-000001"
    assert status.baseline_sequence_number == 0
    assert status.last_event_sequence_number == 1
    assert status.live_document_count == 2
    assert status.baseline_document_count == 1
    assert status.delta_document_count == 1
    assert status.event_count == 1
    assert status.repository_bytes > 0
    assert status.event_log_bytes > 0
    assert status.errors == ()


def test_status_reports_corrupt_manifest_without_raising(tmp_path: Path) -> None:
    repository, event_log = create_repository(tmp_path)
    repository.manifest_path.write_text("not json")

    status = inspect_repository(repository.root, event_log.path)

    assert not status.healthy
    assert status.active_generation is None
    assert status.errors
    assert "RepositoryCorruptionError" in status.errors[0]


def test_status_reports_truncated_wal_tail(tmp_path: Path) -> None:
    repository, event_log = create_repository(tmp_path)
    second = IndexEvent(
        sequence_number=2,
        operation=Operation.DELETE,
        document_id="doc-1",
        version=2,
        event_time=NOW,
    )
    with event_log.path.open("ab") as stream:
        stream.write(encode_record(second)[:-5])

    status = inspect_repository(repository.root, event_log.path)

    assert not status.healthy
    assert status.event_log_truncated_tail
    assert status.live_document_count == 2
    assert "repair is required" in status.errors[0]


def test_runtime_environment_is_populated() -> None:
    environment = RuntimeEnvironment.capture()

    assert environment.python_version
    assert environment.implementation
    assert environment.operating_system
