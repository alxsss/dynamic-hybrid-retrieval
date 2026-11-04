"""Read-only repository validation and operational status reporting."""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dynamic_hybrid_retrieval.compaction import IndexRepository
from dynamic_hybrid_retrieval.event_log import EventLog


def directory_size(path: Path) -> int:
    """Return the total size of regular files below a path."""

    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    python_version: str
    implementation: str
    operating_system: str
    machine: str

    @classmethod
    def capture(cls) -> RuntimeEnvironment:
        return cls(
            python_version=platform.python_version(),
            implementation=platform.python_implementation(),
            operating_system=platform.platform(),
            machine=platform.machine(),
        )


@dataclass(frozen=True, slots=True)
class RepositoryStatus:
    healthy: bool
    checked_at: str
    repository_path: str
    event_log_path: str
    active_generation: str | None
    previous_generation: str | None
    baseline_sequence_number: int | None
    last_event_sequence_number: int | None
    live_document_count: int | None
    baseline_document_count: int | None
    delta_document_count: int | None
    tombstone_count: int | None
    event_count: int | None
    event_log_truncated_tail: bool | None
    repository_bytes: int
    event_log_bytes: int
    recovery_duration_ms: float
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_repository(
    repository_path: str | Path,
    event_log_path: str | Path,
) -> RepositoryStatus:
    """Validate persisted state and simulate normal startup without modifying it."""

    repository_root = Path(repository_path)
    log_path = Path(event_log_path)
    started = time.perf_counter()
    errors: list[str] = []
    active_generation: str | None = None
    previous_generation: str | None = None
    baseline_sequence: int | None = None
    last_event_sequence: int | None = None
    live_documents: int | None = None
    baseline_documents: int | None = None
    delta_documents: int | None = None
    tombstones: int | None = None
    event_count: int | None = None
    truncated_tail: bool | None = None

    try:
        if not repository_root.is_dir():
            raise FileNotFoundError(f"repository directory does not exist: {repository_root}")
        repository = IndexRepository(repository_root)
        manifest = repository.read_manifest()
        active_generation = manifest.active_generation
        previous_generation = manifest.previous_generation
        baseline_sequence = manifest.active_sequence_number

        event_log = EventLog(log_path)
        scan = event_log.scan()
        event_count = len(scan.events)
        truncated_tail = scan.truncated_tail
        last_event_sequence = event_log.last_sequence_number
        if truncated_tail:
            errors.append(
                "event log has an incomplete final record; repair is required before writes resume"
            )

        dynamic = repository.load_active(event_log)
        statistics = dynamic.statistics
        live_documents = statistics.live_document_count
        baseline_documents = statistics.baseline_document_count
        delta_documents = statistics.delta_document_count
        tombstones = statistics.tombstone_count
    except Exception as exc:  # status must convert startup failures into machine-readable output
        errors.append(f"{type(exc).__name__}: {exc}")

    return RepositoryStatus(
        healthy=not errors,
        checked_at=datetime.now(UTC).isoformat(),
        repository_path=str(repository_root.resolve()),
        event_log_path=str(log_path.resolve()),
        active_generation=active_generation,
        previous_generation=previous_generation,
        baseline_sequence_number=baseline_sequence,
        last_event_sequence_number=last_event_sequence,
        live_document_count=live_documents,
        baseline_document_count=baseline_documents,
        delta_document_count=delta_documents,
        tombstone_count=tombstones,
        event_count=event_count,
        event_log_truncated_tail=truncated_tail,
        repository_bytes=directory_size(repository_root),
        event_log_bytes=log_path.stat().st_size if log_path.is_file() else 0,
        recovery_duration_ms=(time.perf_counter() - started) * 1_000,
        errors=tuple(errors),
    )
