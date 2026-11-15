"""Segmented write-ahead log with checkpoint-aware retention."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dynamic_hybrid_retrieval.event_log import (
    AppendResult,
    DuplicateSequenceError,
    EventLog,
    EventLogCorruptionError,
    EventLogError,
    ScanResult,
    SequenceOrderError,
    encode_record,
    event_checksum,
    scan_event_log,
)
from dynamic_hybrid_retrieval.models import IndexEvent

SEGMENTED_WAL_FORMAT_VERSION: Final = 1
ACTIVE_SEGMENT_FILE: Final = "active.jsonl"
CHECKPOINT_FILE: Final = "checkpoint.json"
SEGMENT_PATTERN: Final = re.compile(r"^segment-(\d{20})-(\d{20})\.jsonl$")


class EventLogRetentionError(EventLogError):
    """Raised when recovery requests events that were safely pruned."""


class EventLogCapacityError(EventLogError):
    """Raised when the configured WAL disk budget would be exceeded."""


class CheckpointCorruptionError(EventLogError):
    """Raised when checkpoint metadata cannot be trusted."""


class WalCheckpoint(BaseModel):
    """Durable proof of the sequence range incorporated by active baselines."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    format_version: Annotated[int, Field(ge=1)] = SEGMENTED_WAL_FORMAT_VERSION
    active_generation_sequence: Annotated[int, Field(ge=0)]
    rollback_generation_sequence: Annotated[int, Field(ge=0)] | None = None
    retention_floor_sequence: Annotated[int, Field(ge=0)]
    last_sequence_number: Annotated[int, Field(ge=0)]
    checkpointed_at: datetime

    @model_validator(mode="after")
    def validate_checkpoint(self) -> WalCheckpoint:
        if self.format_version != SEGMENTED_WAL_FORMAT_VERSION:
            raise ValueError(f"unsupported segmented WAL format {self.format_version}")
        expected_floor = min(
            self.active_generation_sequence,
            self.rollback_generation_sequence
            if self.rollback_generation_sequence is not None
            else self.active_generation_sequence,
        )
        if self.retention_floor_sequence != expected_floor:
            raise ValueError("retention floor does not match generation sequences")
        if self.active_generation_sequence > self.last_sequence_number:
            raise ValueError("active generation is ahead of the WAL")
        if (
            self.rollback_generation_sequence is not None
            and self.rollback_generation_sequence > self.last_sequence_number
        ):
            raise ValueError("rollback generation is ahead of the WAL")
        if self.checkpointed_at.tzinfo is None or self.checkpointed_at.utcoffset() is None:
            raise ValueError("checkpointed_at must include a timezone")
        return self


class _CheckpointEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    checkpoint: WalCheckpoint
    checksum: str


def _checkpoint_bytes(checkpoint: WalCheckpoint) -> bytes:
    return orjson.dumps(
        checkpoint.model_dump(mode="json"),
        option=orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
    )


def _checkpoint_checksum(checkpoint: WalCheckpoint) -> str:
    return hashlib.sha256(_checkpoint_bytes(checkpoint)).hexdigest()


class SegmentedEventLog(EventLog):
    """Single-writer WAL composed of immutable sealed segments and one active file.

    The class intentionally implements the existing ``EventLog`` interface so
    dynamic indexes can adopt segmentation without changing replay semantics.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fsync: bool = True,
        max_segment_bytes: int = 64 * 1024 * 1024,
        max_segment_events: int = 100_000,
        max_retained_bytes: int | None = None,
    ) -> None:
        if max_segment_bytes < 1:
            raise ValueError("max_segment_bytes must be positive")
        if max_segment_events < 1:
            raise ValueError("max_segment_events must be positive")
        if max_retained_bytes is not None and max_retained_bytes < 1:
            raise ValueError("max_retained_bytes must be positive")
        self.path = Path(path)
        self.lock_path = self.path / "wal.lock"
        self.active_path = self.path / ACTIVE_SEGMENT_FILE
        self.checkpoint_path = self.path / CHECKPOINT_FILE
        self.fsync = fsync
        self.max_segment_bytes = max_segment_bytes
        self.max_segment_events = max_segment_events
        self.max_retained_bytes = max_retained_bytes
        self._thread_lock = threading.RLock()
        self.path.mkdir(parents=True, exist_ok=True)
        self._events_by_sequence: dict[int, str] = {}
        self._last_sequence = 0
        self._active_event_count = 0
        self._active_size = 0
        self._truncated_tail = False
        self._checkpoint = self._read_checkpoint()
        self._refresh_index()

    @property
    def last_sequence_number(self) -> int:
        return self._last_sequence

    @property
    def retention_floor_sequence(self) -> int:
        return self._checkpoint.retention_floor_sequence if self._checkpoint else 0

    @property
    def checkpoint_state(self) -> WalCheckpoint | None:
        return self._checkpoint

    def _sealed_segments(self) -> list[tuple[int, int, Path]]:
        segments: list[tuple[int, int, Path]] = []
        for candidate in self.path.iterdir():
            match = SEGMENT_PATTERN.fullmatch(candidate.name)
            if match is not None:
                segments.append((int(match.group(1)), int(match.group(2)), candidate))
        return sorted(segments)

    def _read_checkpoint(self) -> WalCheckpoint | None:
        if not self.checkpoint_path.exists():
            return None
        try:
            envelope = _CheckpointEnvelope.model_validate_json(self.checkpoint_path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise CheckpointCorruptionError("WAL checkpoint is missing or invalid") from exc
        if envelope.checksum != _checkpoint_checksum(envelope.checkpoint):
            raise CheckpointCorruptionError("WAL checkpoint checksum mismatch")
        return envelope.checkpoint

    def _write_checkpoint(self, checkpoint: WalCheckpoint) -> None:
        payload = (
            orjson.dumps(
                {
                    "checkpoint": checkpoint.model_dump(mode="json"),
                    "checksum": _checkpoint_checksum(checkpoint),
                },
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_UTC_Z,
            )
            + b"\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".checkpoint-",
            suffix=".tmp",
            dir=self.path,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                if self.fsync:
                    os.fsync(stream.fileno())
            os.replace(temporary_path, self.checkpoint_path)
            self._fsync_directory()
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _fsync_directory(self) -> None:
        if not self.fsync:
            return
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _refresh_index(self) -> ScanResult:
        events: list[IndexEvent] = []
        total_bytes = 0
        previous_sequence = self.retention_floor_sequence
        for start, end, segment_path in self._sealed_segments():
            segment = scan_event_log(segment_path)
            if segment.truncated_tail:
                raise EventLogCorruptionError(
                    segment_path,
                    len(segment.events) + 1,
                    segment.valid_bytes,
                    "sealed segment has a truncated tail",
                )
            if not segment.events or (
                segment.events[0].sequence_number != start
                or segment.events[-1].sequence_number != end
            ):
                raise EventLogCorruptionError(
                    segment_path, 1, 0, "sealed segment filename does not match its events"
                )
            if start <= previous_sequence:
                raise EventLogCorruptionError(
                    segment_path, 1, 0, "segments overlap or are out of order"
                )
            events.extend(segment.events)
            previous_sequence = end
            total_bytes += segment.file_size

        active = scan_event_log(self.active_path)
        if active.events and active.events[0].sequence_number <= previous_sequence:
            raise EventLogCorruptionError(
                self.active_path, 1, 0, "active segment overlaps sealed segments"
            )
        events.extend(active.events)
        total_bytes += active.file_size
        self._events_by_sequence = {
            event.sequence_number: event_checksum(event) for event in events
        }
        retained_last = events[-1].sequence_number if events else 0
        checkpoint_last = self._checkpoint.last_sequence_number if self._checkpoint else 0
        self._last_sequence = max(retained_last, checkpoint_last)
        self._active_event_count = len(active.events)
        self._active_size = active.file_size
        self._truncated_tail = active.truncated_tail
        return ScanResult(
            events=tuple(events),
            truncated_tail=active.truncated_tail,
            valid_bytes=total_bytes - (active.file_size - active.valid_bytes),
            file_size=total_bytes,
        )

    def _seal_active(self) -> Path | None:
        active = scan_event_log(self.active_path)
        if active.truncated_tail:
            raise EventLogError(
                "active WAL segment has a truncated tail; call repair_truncated_tail"
            )
        if not active.events:
            return None
        start = active.events[0].sequence_number
        end = active.events[-1].sequence_number
        sealed_path = self.path / f"segment-{start:020d}-{end:020d}.jsonl"
        if sealed_path.exists():
            raise EventLogError(f"sealed WAL segment already exists: {sealed_path.name}")
        os.replace(self.active_path, sealed_path)
        self._fsync_directory()
        self._active_event_count = 0
        self._active_size = 0
        return sealed_path

    def _retained_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.path.iterdir()
            if path == self.active_path or SEGMENT_PATTERN.fullmatch(path.name) is not None
        )

    def append(self, event: IndexEvent) -> AppendResult:
        """Append to the active segment, rotating before configured boundaries."""

        with self._thread_lock, self.lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                self._checkpoint = self._read_checkpoint()
                self._refresh_index()
                if self._truncated_tail:
                    raise EventLogError(
                        "active WAL segment has a truncated tail; call repair_truncated_tail"
                    )
                checksum = event_checksum(event)
                existing = self._events_by_sequence.get(event.sequence_number)
                if existing is not None:
                    if existing != checksum:
                        raise DuplicateSequenceError(
                            f"sequence {event.sequence_number} already belongs to another event"
                        )
                    return AppendResult(False, event.sequence_number, self._active_size)
                if event.sequence_number <= self.retention_floor_sequence:
                    raise EventLogRetentionError(
                        f"sequence {event.sequence_number} was pruned at retention floor "
                        f"{self.retention_floor_sequence}"
                    )
                if event.sequence_number <= self._last_sequence:
                    raise SequenceOrderError(
                        f"sequence {event.sequence_number} does not advance "
                        f"last sequence {self._last_sequence}"
                    )

                record = encode_record(event)
                if self._active_event_count > 0 and (
                    self._active_event_count >= self.max_segment_events
                    or self._active_size + len(record) > self.max_segment_bytes
                ):
                    self._seal_active()
                if (
                    self.max_retained_bytes is not None
                    and self._retained_bytes() + len(record) > self.max_retained_bytes
                ):
                    raise EventLogCapacityError(
                        "WAL disk budget exceeded; compact and checkpoint before retrying"
                    )

                flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
                file_existed = self.active_path.exists()
                descriptor = os.open(self.active_path, flags, 0o600)
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
                if not file_existed:
                    self._fsync_directory()

                self._events_by_sequence[event.sequence_number] = checksum
                self._last_sequence = event.sequence_number
                self._active_event_count += 1
                self._active_size = byte_offset + len(record)
                return AppendResult(True, event.sequence_number, byte_offset)
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def scan(self) -> ScanResult:
        """Validate and return all retained events across sealed and active segments."""

        with self._thread_lock:
            self._checkpoint = self._read_checkpoint()
            return self._refresh_index()

    def iter_events(self, *, after_sequence: int = 0) -> Iterator[IndexEvent]:
        """Replay retained events, rejecting requests below the retention floor."""

        if after_sequence < self.retention_floor_sequence:
            raise EventLogRetentionError(
                f"cannot replay after sequence {after_sequence}; WAL retention floor is "
                f"{self.retention_floor_sequence}"
            )
        scan = self.scan()
        yield from (event for event in scan.events if event.sequence_number > after_sequence)

    def repair_truncated_tail(self) -> int:
        """Repair only the mutable active segment; sealed corruption is always fatal."""

        with self._thread_lock, self.lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                for _, _, segment_path in self._sealed_segments():
                    sealed = scan_event_log(segment_path)
                    if sealed.truncated_tail:
                        raise EventLogCorruptionError(
                            segment_path,
                            len(sealed.events) + 1,
                            sealed.valid_bytes,
                            "sealed segment has a truncated tail",
                        )
                active = scan_event_log(self.active_path)
                if not active.truncated_tail:
                    self._refresh_index()
                    return 0
                removed = active.file_size - active.valid_bytes
                with self.active_path.open("r+b") as stream:
                    stream.truncate(active.valid_bytes)
                    stream.flush()
                    if self.fsync:
                        os.fsync(stream.fileno())
                self._refresh_index()
                return removed
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def checkpoint(
        self,
        *,
        active_generation_sequence: int,
        rollback_generation_sequence: int | None,
    ) -> WalCheckpoint:
        """Persist a baseline checkpoint, then prune only sealed safe segments."""

        with self._thread_lock, self.lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                self._checkpoint = self._read_checkpoint()
                self._refresh_index()
                retention_floor = min(
                    active_generation_sequence,
                    rollback_generation_sequence
                    if rollback_generation_sequence is not None
                    else active_generation_sequence,
                )
                checkpoint = WalCheckpoint(
                    active_generation_sequence=active_generation_sequence,
                    rollback_generation_sequence=rollback_generation_sequence,
                    retention_floor_sequence=retention_floor,
                    last_sequence_number=self._last_sequence,
                    checkpointed_at=datetime.now(UTC),
                )
                self._seal_active()
                self._write_checkpoint(checkpoint)
                self._checkpoint = checkpoint
                for _, end, segment_path in self._sealed_segments():
                    if end <= retention_floor:
                        segment_path.unlink()
                self._fsync_directory()
                self._refresh_index()
                return checkpoint
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
