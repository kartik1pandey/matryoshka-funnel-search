"""Wall-clock latency benchmarking harness.

"Faster" is a claim that needs a number attached to it, measured on real
hardware, not asserted. This module keeps that measurement consistent
between funnel search and full brute-force search runs.

See docs/04_methodology.md#latency.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyStats:
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    n_trials: int


def benchmark(fn: Callable[[], object], n_trials: int = 20, n_warmup: int = 3) -> LatencyStats:
    """Time repeated calls to `fn`, discarding warm-up runs.

    Args:
        fn: zero-argument callable to time (wrap the real call in a lambda/closure).
        n_trials: number of timed trials.
        n_warmup: number of untimed warm-up calls before timing begins.

    Returns:
        LatencyStats with mean, median, and p95 wall-clock time in seconds.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    for _ in range(n_warmup):
        fn()

    samples = []
    for _ in range(n_trials):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    samples.sort()
    mid = len(samples) // 2
    median = samples[mid] if len(samples) % 2 else (samples[mid - 1] + samples[mid]) / 2
    p95_index = min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))

    return LatencyStats(
        mean_seconds=sum(samples) / len(samples),
        median_seconds=median,
        p95_seconds=samples[p95_index],
        n_trials=n_trials,
    )
