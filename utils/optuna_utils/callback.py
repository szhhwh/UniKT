"""Callbacks integrating Optuna with the training loop."""

import math
from typing import Any

from optuna.trial import Trial

from utils.training import Callback


def _extract_metric_value(name: str, metrics: dict, loss: float | None) -> float | None:
    """Read a named metric from the validation dict (``loss`` from the loss arg).

    Returns None when the metric is absent or non-finite so callers can skip
    the epoch instead of feeding Optuna a placeholder.
    """
    value = loss if name == "loss" else metrics.get(name)
    if value is None or not math.isfinite(value):
        return None
    return float(value)


class OptunaTrialCallback(Callback):
    """Report validation metrics to an Optuna trial and track the best value.

    Supports pruning: sets the ``pruned`` flag and lets the training loop
    end gracefully via ``should_stop`` (preserving resource cleanup in
    ``on_train_end`` / ``_finish``). The caller is expected to raise
    ``TrialPruned`` after ``run()`` returns.

    Multi-stage safe: ``on_train_begin`` fires once per stage, so the per-stage
    best/stop state is reset between stages while ``pruned`` stays sticky
    (once any stage is pruned the whole trial is pruned). ``trial.report`` uses
    a monotonic counter rather than the per-stage epoch, which restarts at 0
    each stage and would otherwise make the step regress across stages.
    """

    def __init__(self, trial: Trial, metric_name: str, maximize: bool):
        """Initialise the Optuna trial callback.

        Args:
            trial: Optuna trial object.
            metric_name: Name of the metric to optimise (case-insensitive).
            maximize: Whether to maximise (True) or minimise (False) the metric.
        """
        self.trial = trial
        self.metric_name = metric_name.lower()
        self.maximize = maximize
        self.best_value: float | None = None
        # Sticky trial-level flag read by the wrapper to raise TrialPruned.
        self.pruned = False
        # Per-stage flag driving should_stop; reset on_train_begin so a prune in
        # an earlier stage does not short-circuit later stages.
        self._stage_pruned = False
        # Globally monotonic report step (the per-stage epoch restarts at 0).
        self._report_step = 0

    def _is_better(self, current: float) -> bool:
        """Check whether the current value is better than the stored best.

        Args:
            current: Current metric value.

        Returns:
            True if the current value is better.
        """
        if self.best_value is None:
            return True
        return current > self.best_value if self.maximize else current < self.best_value

    def on_train_begin(self, epochs: int, **kwargs: Any) -> None:
        """Reset per-stage pruning state.

        ``on_train_begin`` fires once per stage in multi-stage trainers, so this
        isolates each stage's best/stop tracking while leaving the sticky
        ``pruned`` untouched. For single-stage trainers this is a single no-op
        reset of the constructor defaults.

        Args:
            epochs: Total number of epochs in this stage.
            **kwargs: Additional keyword arguments passed by the callback manager.
        """
        self.best_value = None
        self._stage_pruned = False

    def on_phase_end(
        self,
        epoch: int,
        phase: str,
        loss: float,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> None:
        """Handle end-of-phase events: report validation metrics to Optuna.

        Reports the metric value to the trial and checks for pruning.

        Args:
            epoch: Current epoch number.
            phase: Phase name ('train' or 'val').
            loss: Current loss value.
            metrics: Dictionary of current metrics.
            **kwargs: Additional keyword arguments passed by the callback manager.
        """
        if phase != "val":
            return

        value = _extract_metric_value(self.metric_name, metrics, loss)
        if value is None:
            return

        if self._is_better(value):
            self.best_value = value

        self.trial.report(value, self._report_step)
        self._report_step += 1
        if self.trial.should_prune():
            self._stage_pruned = True
            self.pruned = True

    def should_stop(self, **kwargs: Any) -> bool:
        """Check whether the current stage should stop due to pruning.

        Returns:
            True if the current stage has been pruned.
        """
        return self._stage_pruned


class MultiMetricTracker(Callback):
    """Track each objective's own best value across validation epochs.

    Multi-objective search cannot use EarlyStopping's single best-epoch
    snapshot: every objective read from that snapshot is tied to the monitored
    metric's best epoch, flattening the Pareto front. This tracker records each
    metric's independent best so each objective reflects its own optimum.

    Like :class:`OptunaTrialCallback`, it resets per stage (``on_train_begin``)
    so multi-stage trainers expose the final stage's per-metric best.
    """

    def __init__(self, metric_names: list[str], directions: list[str]):
        """Initialise the tracker.

        Args:
            metric_names: Objective metric names (case-insensitive).
            directions: Optimisation direction ("maximize"/"minimize") per
                metric, parallel to ``metric_names``.
        """
        self._names = [n.lower() for n in metric_names]
        self._maximize = {n: d == "maximize" for n, d in zip(self._names, directions)}
        self.best_values: dict[str, float | None] = dict.fromkeys(self._names, None)

    def on_train_begin(self, epochs: int, **kwargs: Any) -> None:
        """Reset per-stage best values.

        Args:
            epochs: Total number of epochs in this stage.
            **kwargs: Additional keyword arguments passed by the callback manager.
        """
        for name in self._names:
            self.best_values[name] = None

    def on_phase_end(
        self,
        epoch: int,
        phase: str,
        loss: float,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> None:
        """Update each metric's best at the end of the validation phase.

        Args:
            epoch: Current epoch number.
            phase: Phase name ('train' or 'val').
            loss: Current loss value.
            metrics: Dictionary of current metrics.
            **kwargs: Additional keyword arguments passed by the callback manager.
        """
        if phase != "val":
            return
        for name in self._names:
            value = _extract_metric_value(name, metrics, loss)
            if value is None:
                continue
            best = self.best_values[name]
            if best is None or (value > best if self._maximize[name] else value < best):
                self.best_values[name] = value


__all__ = ["MultiMetricTracker", "OptunaTrialCallback"]
