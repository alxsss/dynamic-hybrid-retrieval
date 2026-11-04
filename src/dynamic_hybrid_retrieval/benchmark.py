"""End-to-end acceptance benchmark for the baseline-plus-delta index."""

from __future__ import annotations

import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from dynamic_hybrid_retrieval.bm25 import BM25Index, SearchResult
from dynamic_hybrid_retrieval.compaction import IndexRepository
from dynamic_hybrid_retrieval.diagnostics import RuntimeEnvironment, directory_size
from dynamic_hybrid_retrieval.event_log import EventLog
from dynamic_hybrid_retrieval.generate import (
    event_marker,
    generate_documents,
    generate_events,
    generate_queries,
)
from dynamic_hybrid_retrieval.metrics import LatencyRecorder, LatencySummary
from dynamic_hybrid_retrieval.models import Operation


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    document_count: int = 100_000
    event_count: int = 10_000
    query_count: int = 200
    query_limit: int = 10
    seed: int = 42
    freshness_target_ms: float = 1_000.0
    fsync_event_log: bool = True

    def __post_init__(self) -> None:
        for name in ("document_count", "event_count", "query_count", "query_limit"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.freshness_target_ms <= 0:
            raise ValueError("freshness_target_ms must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    config: BenchmarkConfig
    generated_at: str
    environment: RuntimeEnvironment
    baseline_build_seconds: float
    baseline_build_documents_per_second: float
    before_updates_query_latency: LatencySummary
    apply_latency: LatencySummary
    update_visibility_latency: LatencySummary
    after_updates_query_latency: LatencySummary
    replay_seconds: float
    replay_events_per_second: float
    replay_equivalent: bool
    deleted_documents_returned: int
    freshness_violations: int
    compaction_seconds: float
    compaction_equivalent: bool
    final_live_document_count: int
    final_event_sequence: int
    repository_bytes: int
    event_log_bytes: int

    @property
    def passed(self) -> bool:
        return (
            self.replay_equivalent
            and self.deleted_documents_returned == 0
            and self.freshness_violations == 0
            and self.compaction_equivalent
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(orjson.dumps(self.as_dict(), option=orjson.OPT_INDENT_2) + b"\n")
        return destination


def _measure_queries(
    index: Any,
    queries: tuple[str, ...],
    *,
    limit: int,
) -> LatencySummary:
    recorder = LatencyRecorder()
    for query in queries:
        start = time.perf_counter()
        index.search(query, limit=limit)
        recorder.record(time.perf_counter() - start)
    return recorder.summarize()


def _result_signature(results: list[SearchResult]) -> tuple[tuple[str, int, float], ...]:
    return tuple((item.document_id, item.version, item.score) for item in results)


def _queries_equivalent(
    left: Any,
    right: Any,
    queries: tuple[str, ...],
    *,
    limit: int,
) -> bool:
    for query in queries:
        left_results = _result_signature(left.search(query, limit=limit))
        right_results = _result_signature(right.search(query, limit=limit))
        if len(left_results) != len(right_results):
            return False
        for left_item, right_item in zip(left_results, right_results, strict=True):
            if left_item[:2] != right_item[:2] or not math.isclose(
                left_item[2], right_item[2], rel_tol=1e-12, abs_tol=1e-12
            ):
                return False
    return True


def run_benchmark(root: str | Path, config: BenchmarkConfig) -> BenchmarkReport:
    """Run the RFC acceptance workflow in a new, empty directory."""

    benchmark_root = Path(root)
    if benchmark_root.exists() and any(benchmark_root.iterdir()):
        raise FileExistsError(f"benchmark directory must be empty: {benchmark_root}")
    benchmark_root.mkdir(parents=True, exist_ok=True)

    documents = generate_documents(config.document_count, seed=config.seed)
    queries = generate_queries(config.query_count, seed=config.seed + 2)

    build_start = time.perf_counter()
    baseline = BM25Index.build(documents)
    repository = IndexRepository(benchmark_root / "repository")
    repository.initialize(baseline)
    baseline_build_seconds = time.perf_counter() - build_start
    event_log = EventLog(
        benchmark_root / "events.jsonl",
        fsync=config.fsync_event_log,
    )
    dynamic = repository.load_active(event_log)
    before_latency = _measure_queries(dynamic, queries, limit=config.query_limit)

    apply_recorder = LatencyRecorder()
    visibility_recorder = LatencyRecorder()
    deleted_document_ids: set[str] = set()
    marker_by_document: dict[str, str] = {}
    deleted_documents_returned = 0
    for event in generate_events(
        documents,
        config.event_count,
        seed=config.seed + 1,
    ):
        apply_start = time.perf_counter()
        dynamic.apply(event)
        apply_recorder.record(time.perf_counter() - apply_start)

        visibility_start = apply_start
        if event.operation is Operation.UPSERT:
            marker = event_marker(event.sequence_number)
            marker_by_document[event.document_id] = marker
            deleted_document_ids.discard(event.document_id)
            visible = any(
                result.document_id == event.document_id and result.version == event.version
                for result in dynamic.search(marker, limit=1)
            )
            if not visible:
                raise AssertionError(f"upsert {event.sequence_number} was not searchable")
        else:
            deleted_document_ids.add(event.document_id)
            delete_marker = marker_by_document.get(event.document_id)
            if delete_marker is not None and any(
                result.document_id == event.document_id
                for result in dynamic.search(delete_marker, limit=config.query_limit)
            ):
                deleted_documents_returned += 1
        visibility_recorder.record(time.perf_counter() - visibility_start)

    after_latency = _measure_queries(dynamic, queries, limit=config.query_limit)
    deleted_documents_returned += sum(
        result.document_id in deleted_document_ids
        for query in queries
        for result in dynamic.search(query, limit=config.query_limit)
    )
    freshness_summary = visibility_recorder.summarize()
    freshness_violations = sum(
        duration * 1_000 > config.freshness_target_ms
        for duration in visibility_recorder.samples_seconds
    )

    replay_start = time.perf_counter()
    recovered_log = EventLog(event_log.path, fsync=config.fsync_event_log)
    recovered = repository.load_active(recovered_log)
    replay_seconds = time.perf_counter() - replay_start
    replay_equivalent = (
        _queries_equivalent(dynamic, recovered, queries, limit=config.query_limit)
        and dynamic.statistics == recovered.statistics
    )

    expected_results = {
        query: _result_signature(dynamic.search(query, limit=config.query_limit))
        for query in queries
    }
    compaction_start = time.perf_counter()
    repository.compact(
        dynamic,
        validation_queries=queries,
        query_limit=config.query_limit,
    )
    compacted = repository.load_active(event_log)
    compaction_seconds = time.perf_counter() - compaction_start
    compaction_equivalent = all(
        expected_results[query]
        == _result_signature(compacted.search(query, limit=config.query_limit))
        for query in queries
    )

    report = BenchmarkReport(
        config=config,
        generated_at=datetime.now(UTC).isoformat(),
        environment=RuntimeEnvironment.capture(),
        baseline_build_seconds=baseline_build_seconds,
        baseline_build_documents_per_second=config.document_count / baseline_build_seconds,
        before_updates_query_latency=before_latency,
        apply_latency=apply_recorder.summarize(),
        update_visibility_latency=freshness_summary,
        after_updates_query_latency=after_latency,
        replay_seconds=replay_seconds,
        replay_events_per_second=config.event_count / replay_seconds,
        replay_equivalent=replay_equivalent,
        deleted_documents_returned=deleted_documents_returned,
        freshness_violations=freshness_violations,
        compaction_seconds=compaction_seconds,
        compaction_equivalent=compaction_equivalent,
        final_live_document_count=compacted.statistics.live_document_count,
        final_event_sequence=compacted.statistics.last_sequence_number,
        repository_bytes=directory_size(repository.root),
        event_log_bytes=event_log.path.stat().st_size,
    )
    report.write_json(benchmark_root / "report.json")
    return report


def remove_benchmark_directory(path: str | Path) -> None:
    """Explicit cleanup helper for callers that own a benchmark directory."""

    shutil.rmtree(Path(path))
