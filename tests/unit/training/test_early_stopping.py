"""Tests for the EarlyStopping algorithm: strict improvement, patience, mode."""

from typing import Any

import pytest

from utils.config.run_config import EarlyStoppingConfig
from utils.training.early_stopping import EarlyStopping


def _es(**cfg_kwargs: Any) -> EarlyStopping:
    return EarlyStopping(EarlyStoppingConfig(**cfg_kwargs))


class TestConstruction:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'min' or 'max'"):
            _es(mode="sideways")

    def test_case_insensitive_mode(self) -> None:
        es = _es(mode="MIN")
        assert es._cmp_sign == -1.0


class TestStep:
    def test_first_call_initializes_without_bad_epoch(self) -> None:
        es = _es(patience=1)
        assert es.step(0.5, epoch=0) is False
        assert es.best_score == 0.5
        assert es.best_epoch == 0
        assert es.num_bad_epochs == 0

    def test_patience_reached_returns_true(self) -> None:
        es = _es(patience=2, monitor="auc", mode="max")
        es.step(0.9)
        assert es.step(0.8) is False  # bad epoch 1
        assert es.step(0.85) is True  # bad epoch 2 == patience

    def test_improvement_resets_bad_epochs(self) -> None:
        es = _es(patience=2)
        es.step(0.5)
        es.step(0.4)  # bad epoch 1
        es.step(0.6)  # improvement resets
        assert es.num_bad_epochs == 0
        assert es.best_score == 0.6

    def test_strict_improvement_exactly_min_delta_not_improved(self) -> None:
        es = _es(patience=1, min_delta=0.1)
        es.step(0.5)
        # Improving by exactly min_delta is NOT strictly greater -> bad epoch.
        assert es.step(0.6) is True
        assert es.best_score == 0.5

    def test_beyond_min_delta_improves(self) -> None:
        es = _es(min_delta=0.1)
        es.step(0.5)
        es.step(0.6 + 1e-9)
        assert es.best_score == pytest.approx(0.6 + 1e-9)

    def test_min_mode_mirror(self) -> None:
        es = _es(patience=1, mode="min", min_delta=0.0)
        es.step(1.0)
        assert es.step(0.999) is False  # improvement downward
        assert es.best_score == 0.999
        assert es.step(1.5) is True  # bad epoch hits patience 1

    def test_min_mode_with_min_delta(self) -> None:
        es = _es(mode="min", min_delta=0.2)
        es.step(1.0)
        es.step(0.9)  # drop of 0.1 < min_delta 0.2 -> not improved
        assert es.best_score == 1.0
        es.step(0.7)  # drop of 0.3 > min_delta -> improved
        assert es.best_score == 0.7


class TestMetricsSnapshot:
    def test_metrics_copied_defensively(self) -> None:
        es = _es()
        metrics = {"auc": 0.9, "acc": 0.8}
        es.step(0.9, epoch=3, metrics=metrics)
        metrics["auc"] = 0.1  # mutating the caller's dict must not leak in
        assert es.best_metrics == {"auc": 0.9, "acc": 0.8}
        assert es.best_metrics is not metrics

    def test_none_metrics_stored_as_none(self) -> None:
        es = _es()
        es.step(0.5)
        assert es.best_metrics is None
