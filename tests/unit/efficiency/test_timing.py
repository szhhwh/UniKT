"""Tests for ``summarize_latencies``: empty guard, ddof=1 semantics, min/max."""

import pytest

from utils.efficiency.measures.timing import summarize_latencies

_ZEROED = {
    "mean": 0.0,
    "std": 0.0,
    "p50": 0.0,
    "p95": 0.0,
    "p99": 0.0,
    "min": 0.0,
    "max": 0.0,
    "cv": 0.0,
}


class TestSummarizeLatencies:
    def test_empty_input_returns_zeroed_fields(self) -> None:
        assert summarize_latencies([]) == _ZEROED

    def test_single_sample_has_zero_std(self) -> None:
        summary = summarize_latencies([2.0])
        assert summary["mean"] == 2.0
        assert summary["std"] == 0.0
        assert summary["cv"] == 0.0

    def test_two_samples_use_ddof1_std(self) -> None:
        summary = summarize_latencies([1.0, 2.0])
        # population std would be 0.5; ddof=1 gives sqrt(0.5)
        assert summary["std"] == pytest.approx(0.5 * 2**0.5)
        assert summary["cv"] == pytest.approx(summary["std"] / 1.5)

    def test_three_samples_min_max_mean_p50(self) -> None:
        summary = summarize_latencies([3.0, 1.0, 2.0])
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["mean"] == 2.0
        assert summary["p50"] == 2.0
        # ddof=1 on [1,2,3]: sqrt(1.0)
        assert summary["std"] == pytest.approx(1.0)

    def test_keys_are_stable(self) -> None:
        assert set(summarize_latencies([1.0])) == set(_ZEROED)
