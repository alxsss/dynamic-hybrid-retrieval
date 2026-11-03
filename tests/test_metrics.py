from __future__ import annotations

import pytest

from dynamic_hybrid_retrieval.metrics import LatencyRecorder, LatencySummary, percentile


def test_percentile_interpolates_sorted_samples() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0], 95) == pytest.approx(2.9)


def test_percentile_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_summary_converts_seconds_to_milliseconds() -> None:
    summary = LatencySummary.from_seconds([0.001, 0.002, 0.003])

    assert summary.count == 3
    assert summary.minimum_ms == 1.0
    assert summary.mean_ms == 2.0
    assert summary.p95_ms == pytest.approx(2.9)
    assert summary.maximum_ms == 3.0


def test_recorder_rejects_negative_duration() -> None:
    recorder = LatencyRecorder()
    with pytest.raises(ValueError):
        recorder.record(-0.1)
