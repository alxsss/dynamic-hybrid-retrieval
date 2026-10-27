from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamic_hybrid_retrieval.event_log import (
    DuplicateSequenceError,
    EventLog,
    EventLogCorruptionError,
    EventLogError,
    SequenceOrderError,
    encode_record,
    replay_document_state,
    scan_event_log,
)
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def document(document_id: str = "doc-1", version: int = 1, body: str = "body") -> Document:
    return Document(
        document_id=document_id,
        title=f"Title {document_id}",
        body=body,
        updated_at=NOW,
        version=version,
        metadata={"source": "test", "rank": version},
    )


def upsert(sequence: int, *, document_id: str = "doc-1", version: int = 1) -> IndexEvent:
    doc = document(document_id, version, body=f"body-v{version}")
    return IndexEvent(
        sequence_number=sequence,
        operation=Operation.UPSERT,
        document_id=document_id,
        version=version,
        event_time=NOW,
        document=doc,
    )


def delete(sequence: int, *, document_id: str = "doc-1", version: int = 2) -> IndexEvent:
    return IndexEvent(
        sequence_number=sequence,
        operation=Operation.DELETE,
        document_id=document_id,
        version=version,
        event_time=NOW,
    )


def test_models_require_timezone_aware_datetimes() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        Document(
            document_id="doc-1",
            title="Title",
            body="Body",
            updated_at=datetime(2026, 1, 1),
            version=1,
        )


def test_upsert_requires_matching_document_identity_and_version() -> None:
    with pytest.raises(ValidationError, match="IDs must match"):
        IndexEvent(
            sequence_number=1,
            operation=Operation.UPSERT,
            document_id="wrong-id",
            version=1,
            event_time=NOW,
            document=document(),
        )

    with pytest.raises(ValidationError, match="versions must match"):
        IndexEvent(
            sequence_number=1,
            operation=Operation.UPSERT,
            document_id="doc-1",
            version=2,
            event_time=NOW,
            document=document(),
        )


def test_delete_rejects_document_payload() -> None:
    with pytest.raises(ValidationError, match="cannot contain"):
        IndexEvent(
            sequence_number=1,
            operation=Operation.DELETE,
            document_id="doc-1",
            version=1,
            event_time=NOW,
            document=document(),
        )


def test_append_and_scan_round_trip(tmp_path: Path) -> None:
    event_log = EventLog(tmp_path / "events.jsonl")
    events = [upsert(1), upsert(2, version=2), delete(3, version=3)]

    results = [event_log.append(event) for event in events]

    assert all(result.appended for result in results)
    assert list(event_log.iter_events()) == events
    assert list(event_log.iter_events(after_sequence=1)) == events[1:]
    assert event_log.last_sequence_number == 3
    assert not event_log.scan().truncated_tail


def test_exact_duplicate_append_is_idempotent(tmp_path: Path) -> None:
    event_log = EventLog(tmp_path / "events.jsonl")
    event = upsert(1)

    first = event_log.append(event)
    second = event_log.append(event)

    assert first.appended
    assert not second.appended
    assert event_log.scan().events == (event,)


def test_sequence_reuse_with_different_event_is_rejected(tmp_path: Path) -> None:
    event_log = EventLog(tmp_path / "events.jsonl")
    event_log.append(upsert(1))

    with pytest.raises(DuplicateSequenceError):
        event_log.append(upsert(1, document_id="doc-2"))


def test_out_of_order_append_is_rejected(tmp_path: Path) -> None:
    event_log = EventLog(tmp_path / "events.jsonl")
    event_log.append(upsert(2))

    with pytest.raises(SequenceOrderError):
        event_log.append(upsert(1, document_id="doc-2"))


def test_checksum_corruption_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(encode_record(upsert(1)).replace(b"body-v1", b"body-xx"))

    with pytest.raises(EventLogCorruptionError, match="checksum mismatch"):
        scan_event_log(path)


def test_malformed_complete_record_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(encode_record(upsert(1)) + b"not-json\n")

    with pytest.raises(EventLogCorruptionError, match="line 2"):
        scan_event_log(path)


def test_truncated_final_record_is_reported_and_repairable(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    complete = encode_record(upsert(1))
    partial = encode_record(upsert(2, document_id="doc-2"))[:17]
    path.write_bytes(complete + partial)
    event_log = EventLog(path)

    scan = event_log.scan()
    assert scan.events == (upsert(1),)
    assert scan.truncated_tail
    assert scan.file_size > scan.valid_bytes

    with pytest.raises(EventLogError, match="repair_truncated_tail"):
        event_log.append(upsert(3, document_id="doc-3"))

    assert event_log.repair_truncated_tail() == len(partial)
    assert not event_log.scan().truncated_tail
    assert event_log.append(upsert(2, document_id="doc-2")).appended


def test_replay_applies_insert_update_and_delete() -> None:
    events = [
        upsert(1),
        upsert(2, version=2),
        upsert(3, document_id="doc-2"),
        delete(4, version=3),
    ]

    state = replay_document_state(events)

    assert "doc-1" not in state.documents
    assert state.tombstones == {"doc-1": 3}
    assert state.documents["doc-2"].version == 1
    assert state.latest_versions == {"doc-1": 3, "doc-2": 1}
    assert state.applied_events == 4
    assert state.last_sequence_number == 4


def test_replay_ignores_stale_document_versions() -> None:
    events = [upsert(1, version=2), upsert(2, version=1)]

    state = replay_document_state(events)

    assert state.documents["doc-1"].version == 2
    assert state.stale_events == 1
    assert state.applied_events == 1


def test_replay_ignores_exact_duplicate_events() -> None:
    event = upsert(1)

    state = replay_document_state([event, event])

    assert state.applied_events == 1
    assert state.duplicate_events == 1


def test_replay_rejects_conflicting_duplicate_sequence() -> None:
    with pytest.raises(DuplicateSequenceError):
        replay_document_state([upsert(1), upsert(1, document_id="doc-2")])


def test_replay_rejects_decreasing_sequence() -> None:
    with pytest.raises(SequenceOrderError):
        replay_document_state([upsert(2), upsert(1, document_id="doc-2")])
