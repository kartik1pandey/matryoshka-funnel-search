import time

from matryoshka_search.eval.latency import benchmark


def test_benchmark_returns_positive_stats():
    stats = benchmark(lambda: time.sleep(0.001), n_trials=5, n_warmup=1)
    assert stats.mean_seconds > 0
    assert stats.median_seconds > 0
    assert stats.p95_seconds >= stats.median_seconds - 1e-6
    assert stats.n_trials == 5


def test_benchmark_rejects_nonpositive_trials():
    try:
        benchmark(lambda: None, n_trials=0)
        raised = False
    except ValueError:
        raised = True
    assert raised
