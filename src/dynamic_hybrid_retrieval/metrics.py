"""Small dependency-free latency measurement primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def percentile(samples: list[float] | tuple[float, ...], percentile_value: float) -> float:
    """Return a linearly interpolated percentile for non-empty samples."""

    if not samples:
        raise ValueError("at least one sample is required")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between zero and 100")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    minimum_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float

    @classmethod
    def from_seconds(cls, samples: list[float]) -> LatencySummary:
        if not samples:
            raise ValueError("at least one latency sample is required")
        milliseconds = [sample * 1_000 for sample in samples]
        return cls(
            count=len(milliseconds),
            minimum_ms=min(milliseconds),
            mean_ms=sum(milliseconds) / len(milliseconds),
            p50_ms=percentile(milliseconds, 50),
            p95_ms=percentile(milliseconds, 95),
            p99_ms=percentile(milliseconds, 99),
            maximum_ms=max(milliseconds),
        )


@dataclass(slots=True)
class LatencyRecorder:
    """Collect monotonic-duration samples and produce an immutable summary."""

    samples_seconds: list[float] = field(default_factory=list)

    def record(self, duration_seconds: float) -> None:
        if duration_seconds < 0:
            raise ValueError("duration cannot be negative")
        self.samples_seconds.append(duration_seconds)

    def summarize(self) -> LatencySummary:
        return LatencySummary.from_seconds(self.samples_seconds)
