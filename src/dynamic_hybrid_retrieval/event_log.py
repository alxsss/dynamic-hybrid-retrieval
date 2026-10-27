"""Durable append-only event log and deterministic document-state replay."""

from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import orjson
from pydantic import ValidationError

from dynamic_hybrid_retrieval.models import Document, IndexEvent, Operation

RECORD_VERSION: Final = 1


class EventLogError(RuntimeError):
    """Base class for event-log failures."""


class EventLogCorruptionError(EventLogError):
    """Raised when a complete event-log record cannot be trusted."""

    def __init__(self, path: Path, line_number: int, byte_offset: int, reason: str) -> None:
        self.path = path
        self.line_number = line_number
        self.byte_offset = byte_offset
        self.reason = reason
        super().__init__(
            f"corrupt event log {path} at line {line_number}, byte {byte_offset}: {reason}"
        )


class SequenceOrderError(EventLogError):
    """Raised when a new event does not advance the global event sequence."""


class DuplicateSequenceError(EventLogError):
    """Raised when one sequence number refers to two different events."""


@dataclass(frozen=True, slots=True)
class AppendResult:
    """Result of an idempotent append attempt."""

    appended: bool
    sequence_number: int
    byte_offset: int


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Validated records and recovery metadata from one complete log scan."""

    events: tuple[IndexEvent, ...]
    truncated_tail: bool
    valid_bytes: int
    file_size: int


@dataclass(slots=True)
class ReplayState:
    """Logical document state reconstructed from accepted events."""

    documents: dict[str, Document] = field(default_factory=dict)
    tombstones: dict[str, int] = field(default_factory=dict)
    latest_versions: dict[str, int] = field(default_factory=dict)
    last_sequence_number: int = 0
    applied_events: int = 0
    duplicate_events: int = 0
    stale_events: int = 0


def canonical_event_bytes(event: IndexEvent) -> bytes:
    """Return stable bytes used for both persistence and checksumming."""

    return orjson.dumps(
        event.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
    )


def event_checksum(event: IndexEvent) -> str:
    """Return the SHA-256 checksum of an event's canonical representation."""

    return hashlib.sha256(canonical_event_bytes(event)).hexdigest()


def encode_record(event: IndexEvent) -> bytes:
    """Encode an event in the versioned checksummed on-disk record format."""

    record = {
        "record_version": RECORD_VERSION,
        "event": event.model_dump(mode="json"),
        "checksum": event_checksum(event),
    }
    return orjson.dumps(record, option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z) + b"\n"


def _decode_record(data: bytes, path: Path, line_number: int, byte_offset: int) -> IndexEvent:
    try:
        record: Any = orjson.loads(data)
    except orjson.JSONDecodeError as exc:
        raise EventLogCorruptionError(path, line_number, byte_offset, "invalid JSON") from exc

    if not isinstance(record, dict):
        raise EventLogCorruptionError(path, line_number, byte_offset, "record is not an object")
    if record.get("record_version") != RECORD_VERSION:
        raise EventLogCorruptionError(
            path,
            line_number,
            byte_offset,
            f"unsupported record version {record.get('record_version')!r}",
        )
    if set(record) != {"record_version", "event", "checksum"}:
        raise EventLogCorruptionError(path, line_number, byte_offset, "unexpected record fields")

    try:
        event = IndexEvent.model_validate_json(orjson.dumps(record["event"]))
    except (KeyError, ValidationError) as exc:
        raise EventLogCorruptionError(path, line_number, byte_offset, "invalid event") from exc

    checksum = record.get("checksum")
    if not isinstance(checksum, str) or checksum != event_checksum(event):
        raise EventLogCorruptionError(path, line_number, byte_offset, "checksum mismatch")
    return event


def scan_event_log(path: str | Path) -> ScanResult:
    """Validate a log, ignoring only an unterminated final crash fragment.

    Corruption in any newline-terminated record is fatal. A final record with no
    newline is treated as a crash tail and excluded from recovery.
    """

    log_path = Path(path)
    if not log_path.exists():
        return ScanResult(events=(), truncated_tail=False, valid_bytes=0, file_size=0)

    events: list[IndexEvent] = []
    seen: dict[int, str] = {}
    byte_offset = 0
    valid_bytes = 0
    truncated_tail = False

    with log_path.open("rb") as stream:
        line_number = 0
        while line := stream.readline():
            line_number += 1
            start_offset = byte_offset
            byte_offset += len(line)
            if not line.endswith(b"\n"):
                truncated_tail = True
                break

            event = _decode_record(line.removesuffix(b"\n"), log_path, line_number, start_offset)
            checksum = event_checksum(event)
            existing = seen.get(event.sequence_number)
            if existing is not None:
                if existing != checksum:
                    raise EventLogCorruptionError(
                        log_path,
                        line_number,
                        start_offset,
                        "sequence number reused by a different event",
                    )
                raise EventLogCorruptionError(
                    log_path,
                    line_number,
                    start_offset,
                    "duplicate persisted event",
                )
            if events and event.sequence_number <= events[-1].sequence_number:
                raise EventLogCorruptionError(
                    log_path,
                    line_number,
                    start_offset,
                    "event sequence is not strictly increasing",
                )
            seen[event.sequence_number] = checksum
            events.append(event)
            valid_bytes = byte_offset

    return ScanResult(
        events=tuple(events),
        truncated_tail=truncated_tail,
        valid_bytes=valid_bytes,
        file_size=log_path.stat().st_size,
    )


class EventLog:
    """Single-host append-only log with durable, idempotent appends.

    A thread lock protects callers in one process and ``flock`` serializes
    cooperating processes. The project currently assumes one writer, but the
    file lock prevents accidental interleaving by maintenance commands.
    """

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.fsync = fsync
        self._thread_lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events_by_sequence: dict[int, str] = {}
        self._last_sequence = 0
        self._known_file_size = 0
        self._truncated_tail = False
        self._refresh_index()

    @property
    def last_sequence_number(self) -> int:
        return self._last_sequence

    def _refresh_index(self) -> ScanResult:
        scan = scan_event_log(self.path)
        self._events_by_sequence = {
            event.sequence_number: event_checksum(event) for event in scan.events
        }
        self._last_sequence = scan.events[-1].sequence_number if scan.events else 0
        self._known_file_size = scan.file_size
        self._truncated_tail = scan.truncated_tail
        return scan

    def _refresh_if_changed(self) -> ScanResult | None:
        current_size = self.path.stat().st_size if self.path.exists() else 0
        if current_size == self._known_file_size:
            return None
        return self._refresh_index()

    def append(self, event: IndexEvent) -> AppendResult:
        """Durably append an event or return the existing exact duplicate."""

        with self._thread_lock, self.lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                scan = self._refresh_if_changed()
                if self._truncated_tail or (scan is not None and scan.truncated_tail):
                    raise EventLogError(
                        "event log has a truncated tail; call repair_truncated_tail "
                        "before appending"
                    )

                checksum = event_checksum(event)
                existing = self._events_by_sequence.get(event.sequence_number)
                if existing is not None:
                    if existing != checksum:
                        raise DuplicateSequenceError(
                            f"sequence {event.sequence_number} already belongs to another event"
                        )
                    return AppendResult(
                        appended=False,
                        sequence_number=event.sequence_number,
                        byte_offset=self.path.stat().st_size,
                    )

                if event.sequence_number <= self._last_sequence:
                    raise SequenceOrderError(
                        f"sequence {event.sequence_number} does not advance "
                        f"last sequence {self._last_sequence}"
                    )

                record = encode_record(event)
                flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
                file_existed = self.path.exists()
                descriptor = os.open(self.path, flags, 0o600)
                try:
                    byte_offset = os.lseek(descriptor, 0, os.SEEK_END)
                    remaining = memoryview(record)
                    while remaining:
                        written = os.write(descriptor, remaining)
                        if written == 0:
                            raise EventLogError("zero-byte write while appending event")
                        remaining = remaining[written:]
                    if self.fsync:
                        os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                if self.fsync and not file_existed:
                    directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_descriptor)
                    finally:
                        os.close(directory_descriptor)

                self._events_by_sequence[event.sequence_number] = checksum
                self._last_sequence = event.sequence_number
                self._known_file_size = byte_offset + len(record)
                return AppendResult(
                    appended=True,
                    sequence_number=event.sequence_number,
                    byte_offset=byte_offset,
                )
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def scan(self) -> ScanResult:
        """Return a fully validated snapshot of the log."""

        with self._thread_lock:
            return self._refresh_index()

    def iter_events(self, *, after_sequence: int = 0) -> Iterator[IndexEvent]:
        """Iterate over a validated snapshot after an exclusive sequence bound."""

        scan = self.scan()
        yield from (event for event in scan.events if event.sequence_number > after_sequence)

    def repair_truncated_tail(self) -> int:
        """Remove an incomplete final record and return the number of bytes removed."""

        with self._thread_lock, self.lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                scan = scan_event_log(self.path)
                if not scan.truncated_tail:
                    self._refresh_index()
                    return 0
                removed = scan.file_size - scan.valid_bytes
                with self.path.open("r+b") as stream:
                    stream.truncate(scan.valid_bytes)
                    stream.flush()
                    if self.fsync:
                        os.fsync(stream.fileno())
                self._refresh_index()
                return removed
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def replay_document_state(events: Iterable[IndexEvent]) -> ReplayState:
    """Apply ordered events into a deterministic latest-version state."""

    state = ReplayState()
    seen_sequences: dict[int, str] = {}

    for event in events:
        checksum = event_checksum(event)
        existing = seen_sequences.get(event.sequence_number)
        if existing is not None:
            if existing != checksum:
                raise DuplicateSequenceError(
                    f"sequence {event.sequence_number} belongs to different events"
                )
            state.duplicate_events += 1
            continue
        if event.sequence_number <= state.last_sequence_number:
            raise SequenceOrderError(
                f"sequence {event.sequence_number} follows {state.last_sequence_number}"
            )

        seen_sequences[event.sequence_number] = checksum
        state.last_sequence_number = event.sequence_number
        latest_version = state.latest_versions.get(event.document_id, 0)
        if event.version <= latest_version:
            state.stale_events += 1
            continue

        state.latest_versions[event.document_id] = event.version
        if event.operation is Operation.UPSERT:
            assert event.document is not None
            state.documents[event.document_id] = event.document
            state.tombstones.pop(event.document_id, None)
        else:
            state.documents.pop(event.document_id, None)
            state.tombstones[event.document_id] = event.version
        state.applied_events += 1

    return state
