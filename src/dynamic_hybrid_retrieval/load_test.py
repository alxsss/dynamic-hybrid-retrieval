"""Concurrent hybrid retrieval load and recovery verification."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from dynamic_hybrid_retrieval.bm25 import BM25Index
from dynamic_hybrid_retrieval.dense import HNSWIndex
from dynamic_hybrid_retrieval.embeddings import HashingEmbedding
from dynamic_hybrid_retrieval.generate import (
    event_marker,
    generate_documents,
    generate_events,
    generate_queries,
)
from dynamic_hybrid_retrieval.hybrid_repository import HybridIndexRepository
from dynamic_hybrid_retrieval.metrics import LatencyRecorder, LatencySummary
from dynamic_hybrid_retrieval.models import Operation
from dynamic_hybrid_retrieval.segmented_event_log import SegmentedEventLog


@dataclass(frozen=True, slots=True)
class LoadTestConfig:
    documents: int = 1_000
    events: int = 200
    reader_threads: int = 4
    queries_per_reader: int = 100
    seed: int = 42
    freshness_target_ms: float = 1_000


@dataclass(frozen=True, slots=True)
class LoadTestReport:
    generated_at: str
    config: LoadTestConfig
    query_latency: LatencySummary
    apply_latency: LatencySummary
    freshness_latency: LatencySummary
    compaction_seconds: float
    recovery_seconds: float
    queries_completed: int
    events_applied: int
    freshness_violations: int
    deleted_documents_returned: int
    worker_errors: tuple[str, ...]
    recovered_equivalent: bool

    @property
    def passed(self) -> bool:
        return (
            not self.worker_errors
            and not self.freshness_violations
            and not self.deleted_documents_returned
            and self.recovered_equivalent
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


def run_load_test(root: str | Path, config: LoadTestConfig) -> LoadTestReport:
    target = Path(root)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"load-test directory must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    documents = generate_documents(config.documents, seed=config.seed)
    embedding = HashingEmbedding(64)
    repository = HybridIndexRepository(target / "repository")
    repository.initialize(BM25Index.build(documents), HNSWIndex.build(documents, embedding))
    wal = SegmentedEventLog(target / "wal", fsync=False, max_segment_events=25)
    hybrid = repository.load_active(wal, embedding)
    events = tuple(generate_events(documents, config.events, seed=config.seed + 1))
    queries = generate_queries(config.queries_per_reader, seed=config.seed + 2)
    query_times, apply_times, freshness_times = (
        LatencyRecorder(),
        LatencyRecorder(),
        LatencyRecorder(),
    )
    errors: list[str] = []
    deleted_returned = 0
    midpoint = threading.Event()

    def reader() -> None:
        try:
            for query in queries:
                started = time.perf_counter()
                hybrid.search(query, limit=10)
                query_times.record(time.perf_counter() - started)
        except Exception as exc:
            errors.append(repr(exc))

    def writer() -> None:
        nonlocal deleted_returned
        try:
            for position, event in enumerate(events):
                started = time.perf_counter()
                hybrid.apply(event)
                apply_times.record(time.perf_counter() - started)
                if event.operation is Operation.UPSERT:
                    visible = any(
                        item.document_id == event.document_id and item.version == event.version
                        for item in hybrid.search(event_marker(event.sequence_number), limit=1)
                    )
                    if not visible:
                        errors.append(f"event {event.sequence_number} not visible")
                else:
                    deleted_returned += sum(
                        item.document_id == event.document_id
                        for item in hybrid.search(event_marker(event.sequence_number), limit=10)
                    )
                freshness_times.record(time.perf_counter() - started)
                if position + 1 == max(1, len(events) // 2):
                    midpoint.set()
        except Exception as exc:
            errors.append(repr(exc))
            midpoint.set()

    compaction_duration = 0.0

    def compact() -> None:
        nonlocal compaction_duration
        midpoint.wait()
        started = time.perf_counter()
        try:
            repository.compact(hybrid, validation_queries=queries[:10])
        except Exception as exc:
            errors.append(repr(exc))
        compaction_duration = time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=config.reader_threads + 2) as pool:
        futures = [pool.submit(reader) for _ in range(config.reader_threads)] + [
            pool.submit(writer),
            pool.submit(compact),
        ]
        for future in futures:
            future.result()

    recovery_start = time.perf_counter()
    recovered = repository.load_active(SegmentedEventLog(target / "wal", fsync=False), embedding)
    recovery_seconds = time.perf_counter() - recovery_start

    def signatures(index: Any) -> list[list[tuple[str, int]]]:
        return [
            [
                (result.document_id, result.version)
                for result in index.search(query, limit=10, exact_dense=True)
            ]
            for query in queries[:20]
        ]

    equivalent = (
        signatures(hybrid) == signatures(recovered)
        and hybrid.statistics.last_sequence_number == recovered.statistics.last_sequence_number
    )
    violations = sum(
        value * 1000 > config.freshness_target_ms for value in freshness_times.samples_seconds
    )
    report = LoadTestReport(
        datetime.now(UTC).isoformat(),
        config,
        query_times.summarize(),
        apply_times.summarize(),
        freshness_times.summarize(),
        compaction_duration,
        recovery_seconds,
        config.reader_threads * len(queries),
        len(events),
        violations,
        deleted_returned,
        tuple(errors),
        equivalent,
    )
    (target / "report.json").write_bytes(
        orjson.dumps(report.as_dict(), option=orjson.OPT_INDENT_2) + b"\n"
    )
    return report
