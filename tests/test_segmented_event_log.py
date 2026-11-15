from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dynamic_hybrid_retrieval.event_log import (
    DuplicateSequenceError,
    EventLogCorruptionError,
    EventLogError,
    SequenceOrderError,
    encode_record,
)
from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation
from dynamic_hybrid_retrieval.segmented_event_log import (
    ACTIVE_SEGMENT_FILE,
    CheckpointCorruptionError,
    EventLogCapacityError,
    EventLogRetentionError,
    SegmentedEventLog,
)

NOW = datetime(2026, 11, 14, tzinfo=UTC)


def upsert(sequence: int, document_id: str | None = None) -> IndexEvent:
    active_id = document_id or f"doc-{sequence}"
    document = Document(
        document_id=active_id,
        title=f"Document {sequence}",
        body=f"body for sequence {sequence}",
        updated_at=NOW,
        version=1,
    )
    return IndexEvent(
        sequence_number=sequence,
        operation=Operation.UPSERT,
        document_id=active_id,
        version=1,
        event_time=NOW,
        document=document,
    )


def segmented(tmp_path: Path, **kwargs: int | bool | None) -> SegmentedEventLog:
    return SegmentedEventLog(
        tmp_path / "wal",
        fsync=False,
        max_segment_events=2,
        **kwargs,
    )


def test_rotation_and_restart_recover_all_segments(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    events = [upsert(sequence) for sequence in range(1, 6)]
    for event in events:
        log.append(event)

    assert [path.name for path in (tmp_path / "wal").glob("segment-*.jsonl")] == [
        "segment-00000000000000000001-00000000000000000002.jsonl",
        "segment-00000000000000000003-00000000000000000004.jsonl",
    ]
    assert list(log.iter_events()) == events

    recovered = segmented(tmp_path)
    assert list(recovered.iter_events()) == events
    assert recovered.last_sequence_number == 5


def test_duplicate_and_ordering_rules_span_segments(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    first = upsert(1)
    log.append(first)
    log.append(upsert(2))
    log.append(upsert(3))

    assert not log.append(first).appended
    with pytest.raises(DuplicateSequenceError):
        log.append(upsert(1, "different"))

    ordered_log = SegmentedEventLog(tmp_path / "ordered-wal", fsync=False)
    ordered_log.append(upsert(2))
    with pytest.raises(SequenceOrderError):
        ordered_log.append(upsert(1, "late"))


def test_truncated_active_tail_is_repairable(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    log.append(upsert(1))
    partial = encode_record(upsert(2))[:19]
    with (log.path / ACTIVE_SEGMENT_FILE).open("ab") as stream:
        stream.write(partial)

    assert log.scan().truncated_tail
    with pytest.raises(EventLogError, match="repair_truncated_tail"):
        log.append(upsert(3))
    assert log.repair_truncated_tail() == len(partial)
    assert log.append(upsert(2)).appended


def test_sealed_corruption_is_fatal_and_never_repaired(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    for sequence in range(1, 4):
        log.append(upsert(sequence))
    sealed = next(log.path.glob("segment-*.jsonl"))
    sealed.write_bytes(sealed.read_bytes()[:-1])

    with pytest.raises(EventLogCorruptionError, match="sealed segment"):
        SegmentedEventLog(log.path, fsync=False)
    with pytest.raises(EventLogCorruptionError, match="sealed segment"):
        log.repair_truncated_tail()


def test_restart_after_rotation_without_active_file(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    log.append(upsert(1))
    log.append(upsert(2))
    log._seal_active()
    assert not log.active_path.exists()

    recovered = segmented(tmp_path)
    assert list(recovered.iter_events()) == [upsert(1), upsert(2)]
    assert recovered.append(upsert(3)).appended


def test_checkpoint_retains_events_needed_by_rollback(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    for sequence in range(1, 7):
        log.append(upsert(sequence))

    checkpoint = log.checkpoint(
        active_generation_sequence=6,
        rollback_generation_sequence=2,
    )

    assert checkpoint.retention_floor_sequence == 2
    assert [event.sequence_number for event in log.iter_events(after_sequence=2)] == [3, 4, 5, 6]
    with pytest.raises(EventLogRetentionError, match="retention floor"):
        list(log.iter_events(after_sequence=0))
    assert not any(
        "00000000000000000001-00000000000000000002" in path.name for path in log.path.iterdir()
    )


def test_checkpoint_without_rollback_can_prune_every_current_event(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    for sequence in range(1, 5):
        log.append(upsert(sequence))

    log.checkpoint(active_generation_sequence=4, rollback_generation_sequence=None)

    assert log.scan().events == ()
    assert log.last_sequence_number == 4
    assert log.retention_floor_sequence == 4
    assert log.append(upsert(5)).appended
    with pytest.raises(EventLogRetentionError):
        log.append(upsert(4))


def test_checkpoint_corruption_is_detected(tmp_path: Path) -> None:
    log = segmented(tmp_path)
    log.append(upsert(1))
    log.checkpoint(active_generation_sequence=1, rollback_generation_sequence=None)
    log.checkpoint_path.write_bytes(
        log.checkpoint_path.read_bytes().replace(
            b'"last_sequence_number": 1', b'"last_sequence_number": 9'
        )
    )

    with pytest.raises(CheckpointCorruptionError, match="checksum"):
        SegmentedEventLog(log.path, fsync=False)


def test_capacity_limit_rejects_write_without_partial_record(tmp_path: Path) -> None:
    record_size = len(encode_record(upsert(1)))
    log = segmented(tmp_path, max_retained_bytes=record_size)
    log.append(upsert(1))

    with pytest.raises(EventLogCapacityError, match="disk budget"):
        log.append(upsert(2))

    assert list(log.iter_events()) == [upsert(1)]
