from pathlib import Path

from dynamic_hybrid_retrieval.load_test import LoadTestConfig, run_load_test


def test_concurrent_load_test_passes(tmp_path: Path) -> None:
    report = run_load_test(
        tmp_path / "load",
        LoadTestConfig(documents=40, events=10, reader_threads=2, queries_per_reader=8),
    )
    assert report.passed
    assert report.queries_completed == 16
    assert report.events_applied == 10
    assert report.query_latency.p99_ms >= 0
    assert report.recovery_seconds >= 0
