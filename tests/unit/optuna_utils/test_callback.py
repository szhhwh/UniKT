"""Tests for Optuna callbacks: metric extraction, pruning, per-metric tracking."""

import math
from collections.abc import Callable
from typing import Any

from utils.optuna_utils.callback import (
    MultiMetricTracker,
    OptunaTrialCallback,
    _extract_metric_value,
)

# --- _extract_metric_value ---


class TestExtractMetricValue:
    def test_loss_alias_reads_from_loss_arg(self) -> None:
        assert _extract_metric_value("loss", {"auc": 0.7}, 0.25) == 0.25

    def test_named_metric_reads_from_metrics(self) -> None:
        assert _extract_metric_value("auc", {"auc": 0.7}, None) == 0.7

    def test_missing_metric_returns_none(self) -> None:
        assert _extract_metric_value("auc", {"rmse": 0.2}, None) is None

    def test_nan_returns_none(self) -> None:
        assert _extract_metric_value("auc", {"auc": math.nan}, None) is None

    def test_inf_returns_none(self) -> None:
        assert _extract_metric_value("loss", {}, math.inf) is None

    def test_result_is_coerced_to_float(self) -> None:
        assert _extract_metric_value("auc", {"auc": 1}, None) == 1.0
        assert isinstance(_extract_metric_value("auc", {"auc": 1}, None), float)


# --- OptunaTrialCallback ---


def _phase_end(
    cb: OptunaTrialCallback | MultiMetricTracker,
    epoch: int,
    metrics: dict,
    loss: float = 0.5,
    phase: str = "val",
) -> None:
    cb.on_phase_end(epoch=epoch, phase=phase, loss=loss, metrics=metrics)


class TestOptunaTrialCallback:
    def test_reports_every_val_epoch_with_monotonic_step(self, fake_trial: Any) -> None:
        cb = OptunaTrialCallback(fake_trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": 0.6})
        _phase_end(cb, 1, {"auc": 0.7})
        assert fake_trial.reports == [(0.6, 0), (0.7, 1)]
        assert cb.best_value == 0.7

    def test_best_tracks_maximize_direction(self, fake_trial: Any) -> None:
        cb = OptunaTrialCallback(fake_trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": 0.7})
        _phase_end(cb, 1, {"auc": 0.5})  # worse: reported, best unchanged
        assert cb.best_value == 0.7
        assert len(fake_trial.reports) == 2

    def test_best_tracks_minimize_direction(self, fake_trial: Any) -> None:
        cb = OptunaTrialCallback(fake_trial, "loss", maximize=False)
        _phase_end(cb, 0, {"auc": 1.0}, loss=0.9)
        _phase_end(cb, 1, {"auc": 1.0}, loss=0.4)
        _phase_end(cb, 2, {"auc": 1.0}, loss=0.6)
        assert cb.best_value == 0.4

    def test_train_phase_is_ignored(self, fake_trial: Any) -> None:
        cb = OptunaTrialCallback(fake_trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": 0.9}, phase="train")
        assert fake_trial.reports == []
        assert cb.best_value is None

    def test_non_finite_value_skips_report(self, fake_trial: Any) -> None:
        cb = OptunaTrialCallback(fake_trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": math.nan})
        assert fake_trial.reports == []
        assert cb.best_value is None

    def test_prune_sets_stage_and_trial_flags(
        self, make_fake_trial: Callable[..., Any]
    ) -> None:
        trial = make_fake_trial(should_prune=True)
        cb = OptunaTrialCallback(trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": 0.6})
        assert cb.pruned is True
        assert cb.should_stop() is True

    def test_pruned_is_sticky_across_stages(
        self, make_fake_trial: Callable[..., Any]
    ) -> None:
        trial = make_fake_trial(should_prune=True)
        cb = OptunaTrialCallback(trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": 0.6})
        cb.on_train_begin(epochs=3)
        assert cb.pruned is True  # sticky
        assert cb.should_stop() is False  # per-stage flag reset

    def test_on_train_begin_resets_best(self, fake_trial: Any) -> None:
        cb = OptunaTrialCallback(fake_trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": 0.9})
        cb.on_train_begin(epochs=2)
        assert cb.best_value is None

    def test_report_step_is_global_across_stages(self, fake_trial: Any) -> None:
        cb = OptunaTrialCallback(fake_trial, "auc", maximize=True)
        _phase_end(cb, 0, {"auc": 0.6})
        _phase_end(cb, 1, {"auc": 0.7})
        cb.on_train_begin(epochs=1)  # stage two: epoch restarts at 0
        _phase_end(cb, 0, {"auc": 0.8})
        assert [step for _, step in fake_trial.reports] == [0, 1, 2]


# --- MultiMetricTracker ---


class TestMultiMetricTracker:
    def test_independent_per_metric_bests(self) -> None:
        tracker = MultiMetricTracker(["auc", "rmse"], ["maximize", "minimize"])
        _tracker_phase(tracker, {"auc": 0.6, "rmse": 0.5})
        _tracker_phase(tracker, {"auc": 0.7, "rmse": 0.4})
        _tracker_phase(tracker, {"auc": 0.5, "rmse": 0.6})
        assert tracker.best_values == {"auc": 0.7, "rmse": 0.4}

    def test_missing_metric_stays_none(self) -> None:
        tracker = MultiMetricTracker(["auc", "rmse"], ["maximize", "minimize"])
        _tracker_phase(tracker, {"auc": 0.6})
        assert tracker.best_values["auc"] == 0.6
        assert tracker.best_values["rmse"] is None

    def test_train_phase_ignored(self) -> None:
        tracker = MultiMetricTracker(["auc"], ["maximize"])
        tracker.on_phase_end(epoch=0, phase="train", loss=0.5, metrics={"auc": 0.9})
        assert tracker.best_values == {"auc": None}

    def test_on_train_begin_resets_bests(self) -> None:
        tracker = MultiMetricTracker(["auc"], ["maximize"])
        _tracker_phase(tracker, {"auc": 0.6})
        tracker.on_train_begin(epochs=1)
        assert tracker.best_values == {"auc": None}

    def test_loss_objective_read_from_loss_arg(self) -> None:
        tracker = MultiMetricTracker(["loss"], ["minimize"])
        tracker.on_phase_end(epoch=0, phase="val", loss=0.3, metrics={})
        tracker.on_phase_end(epoch=1, phase="val", loss=0.2, metrics={})
        assert tracker.best_values == {"loss": 0.2}


def _tracker_phase(tracker: MultiMetricTracker, metrics: dict) -> None:
    tracker.on_phase_end(epoch=0, phase="val", loss=0.5, metrics=metrics)
