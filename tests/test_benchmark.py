from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from dynamic_hybrid_retrieval.benchmark import BenchmarkConfig, run_benchmark


def test_small_benchmark_exercises_acceptance_workflow(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    report = run_benchmark(
        output,
        BenchmarkConfig(
            document_count=30,
            event_count=10,
            query_count=8,
            fsync_event_log=False,
        ),
    )

    assert report.passed
    assert report.before_updates_query_latency.count == 8
    assert report.after_updates_query_latency.count == 8
    assert report.apply_latency.count == 10
    assert report.update_visibility_latency.count == 10
    assert report.final_event_sequence == 10
    assert orjson.loads((output / "report.json").read_bytes())["passed"] is True


def test_benchmark_refuses_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "benchmark"
    output.mkdir()
    (output / "keep.txt").write_text("user data")

    with pytest.raises(FileExistsError):
        run_benchmark(output, BenchmarkConfig(document_count=1, event_count=1, query_count=1))
