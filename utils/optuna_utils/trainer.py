"""Trainer integration with Optuna objective functions."""

from __future__ import annotations

import gc
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from utils.core import add_file_handler, get_logger, seed_everything

if TYPE_CHECKING:
    from utils.experiment_manager import ExperimentManager

from .callback import MultiMetricTracker, OptunaTrialCallback
from .config import direction_for_metric

logger = get_logger(__name__)


class TrainerObjectiveWrapper:
    """Wrapper integrating a Trainer into an Optuna objective function."""

    def __init__(
        self,
        trainer_class: type,
        data_src_fn: Callable[[], Any],
        base_rc: Any,
        metric_name: str | list[str] = "auc",
        exp_manager: ExperimentManager | None = None,
    ):
        """Initialise the Trainer wrapper.

        Args:
            trainer_class: Trainer class.
            data_src_fn: Data source factory function.
            base_rc: Base RunConfig instance.
            metric_name: Metric name (str) or names (list) to optimise; a list
                enables multi-objective search.
            exp_manager: Experiment manager for creating trial subdirectories.
        """
        self.trainer_class = trainer_class
        self.data_src_fn = data_src_fn
        self.base_rc = base_rc
        self.metric_names = (
            [metric_name] if isinstance(metric_name, str) else list(metric_name)
        )
        self._multi = len(self.metric_names) > 1
        self.maximize = direction_for_metric(self.metric_names[0]) == "maximize"
        # Per-objective directions let the multi-objective tracker keep each
        # metric's own best rather than a shared best-epoch snapshot.
        self._directions = [direction_for_metric(m) for m in self.metric_names]
        self.exp_manager = exp_manager

    def __call__(
        self,
        trial: optuna.trial.Trial,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> float | list[float]:
        """Execute a single trial with a given hyperparameter combination.

        Args:
            trial: Optuna trial object.
            params: Hyperparameter dictionary for this trial (model-node field
                names; applied as a merge overlay onto ``base_rc.model``).
            **kwargs: Additional keyword arguments.

        Returns:
            The optimised metric value for this trial.
        """
        if params is None:
            params = {}

        start_time = time.perf_counter()
        trainer = None

        try:
            trial_rc = self._create_trial_rc(params)

            # Reseed before constructing the trainer so every trial starts from
            # the same RNG state (weight init, data shuffle, training RNG).
            seed_everything(
                trial_rc.general.seed,
                deterministic=trial_rc.general.deterministic,
            )
            trial_exp_manager = None
            if self.exp_manager is not None:
                trial_exp_manager = self.exp_manager.create_sub_experiment(
                    f"trial_{trial.number}"
                )
                # Point the shared file sink at this trial's run.log so each
                # trial dir is self-contained; the parent sink is restored in
                # finally to keep per-trial epoch output out of the search log.
                add_file_handler(Path(trial_exp_manager.get_log_dir()) / "run.log")

            # Pruning is single-objective only (trial.report takes one value);
            # multi-objective instead tracks each objective's own best.
            pruning_cb = None
            tracker = None
            if not self._multi:
                pruning_cb = OptunaTrialCallback(
                    trial=trial,
                    metric_name=self.metric_names[0],
                    maximize=self.maximize,
                )
            else:
                tracker = MultiMetricTracker(self.metric_names, self._directions)

            data_src = self.data_src_fn()
            trainer = self.trainer_class(
                rc=trial_rc, data_src=data_src, exp_manager=trial_exp_manager
            )
            # Register callbacks via the trainer's controlled hook: single-stage
            # trainers append to the live list; multi-stage trainers register
            # into _custom_callbacks so every stage includes it.
            if pruning_cb is not None:
                trainer.add_callback(pruning_cb)
            if tracker is not None:
                trainer.add_callback(tracker)

            # Record the trial dir before run() so it survives a mid-run failure;
            # OptunaTuner copies the best trial's run_config.yaml from here.
            if trial_exp_manager is not None:
                trial.set_user_attr("trial_dir", trial_exp_manager.get_log_dir())

            trainer.run()

            # Record attrs before any branch that raises, so pruned trials keep
            # best_epoch/duration_sec in search_history.yaml instead of nulls.
            es = getattr(trainer, "early_stopping", None)
            best_epoch = es.best_epoch if es is not None else None
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr(
                "duration_sec", round(time.perf_counter() - start_time, 2)
            )

            if pruning_cb is not None and pruning_cb.pruned:
                raise optuna.TrialPruned(
                    f"Trial {trial.number} pruned at epoch {best_epoch}"
                )

            return self._extract_metric(trainer, pruning_cb, tracker)

        except optuna.TrialPruned:
            # Pruning is intentional, not a failure — keep the trial PRUNED.
            raise

        except Exception as e:
            # Record the root cause for debugging; re-raise so
            # study.optimize(catch=...) marks the trial FAIL (not COMPLETE).
            logger.exception(f"Trial {trial.number} failed")
            trial.set_user_attr("error", repr(e))
            trial.set_user_attr("traceback", traceback.format_exc())
            trial.set_user_attr(
                "duration_sec", round(time.perf_counter() - start_time, 2)
            )
            raise

        finally:
            # Restore the parent sink so inter-trial and summary logs land in
            # the search dir rather than the last trial's run.log.
            if self.exp_manager is not None:
                add_file_handler(Path(self.exp_manager.get_log_dir()) / "run.log")
            # Release DataLoader workers deterministically: an exceptional
            # exit (e.g. a pruned trial) keeps traceback frames — and with
            # them the trainer's loaders — alive past the trial, and the
            # leaked worker pipes accumulate up to the process fd limit.
            if trainer is not None:
                try:
                    trainer.release()
                except Exception:
                    logger.warning(
                        "Trainer release failed; worker pipes may leak",
                        exc_info=True,
                    )
            gc.collect()

    def _create_trial_rc(self, params: dict[str, Any]) -> Any:
        """Evolve the base RunConfig with a trial's model hyperparameters.

        Trial params are model-node field names; they are applied onto a deep
        copy of ``base_rc.model``. Every trial reseeds from the base seed so only
        the sampled hyperparameters vary across trials.
        """
        import copy

        trial_rc = copy.deepcopy(self.base_rc)
        # Pin-memory loaders leak ~16 pipe fds per DataLoader on CUDA
        # (torch 2.10), which a multi-trial search exhausts the process
        # fd limit with; auto pinning is therefore disabled per trial.
        if self.base_rc.general.pin_memory is None:
            trial_rc.general.pin_memory = False
        for name, value in params.items():
            setattr(trial_rc.model, name, value)
        return trial_rc

    def _extract_metric(
        self,
        trainer: Any,
        pruning_cb: OptunaTrialCallback | None,
        tracker: MultiMetricTracker | None = None,
    ) -> float | list[float]:
        """Extract the optimised metric value(s) from the trainer.

        Single-objective: prefers the pruning callback's tracked best value,
        falling back to the EarlyStopping best-epoch metrics snapshot.
        Multi-objective: reads each objective's own best from the per-metric
        tracker.

        Args:
            trainer: The trainer instance.
            pruning_cb: The Optuna trial callback (None for multi-objective).
            tracker: The multi-objective per-metric tracker (None for
                single-objective).

        Returns:
            The metric value (single-objective) or a list of values
            (multi-objective).
        """
        if self._multi:
            return self._extract_multi(tracker)

        metric_lower = self.metric_names[0].lower()

        # Prefer callback-tracked best value: it reads the raw metric dict with
        # no surrogate substitution, so it is always the genuine metric.
        if pruning_cb is not None and pruning_cb.best_value is not None:
            return pruning_cb.best_value

        es = getattr(trainer, "early_stopping", None)
        if es is not None and es.best_metrics:
            value = es.best_metrics.get(metric_lower)
            if value is not None:
                return float(value)

        # Intentionally do NOT fall back to es.best_score: EarlyStoppingCallback
        # silently substitutes a surrogate (auc->auprc->acc->rmse) when the
        # monitored metric is missing, so best_score may not be the requested
        # metric and reporting it would mislead Optuna.
        raise RuntimeError(
            f"Could not extract metric '{metric_lower}' from trainer "
            f"(pruning_cb.best_value="
            f"{pruning_cb.best_value if pruning_cb is not None else None}, "
            f"early_stopping attached={es is not None}). "
            f"Ensure validation data is provided and the metric name is correct."
        )

    def _extract_multi(self, tracker: MultiMetricTracker | None) -> list[float]:
        """Extract each objective's own best from the per-metric tracker.

        Each objective reflects its independent best across validation epochs
        rather than a shared best-epoch snapshot, preserving the Pareto front
        and decoupling objectives from the unrelated ``early_stopping.monitor``.

        Args:
            tracker: The multi-objective per-metric tracker.

        Returns:
            A list of per-objective best values, parallel to ``metric_names``.
        """
        if tracker is None:
            raise RuntimeError(
                "Multi-objective per-metric tracker was not registered; cannot "
                "extract per-objective values."
            )
        values: list[float] = []
        for name in self.metric_names:
            value = tracker.best_values.get(name.lower())
            if value is None:
                tracked = {
                    k: v for k, v in tracker.best_values.items() if v is not None
                }
                raise RuntimeError(
                    f"Could not extract metric '{name}' from tracker "
                    f"(tracked={tracked or 'none'}). "
                    f"Ensure validation data is provided and the metric name is correct."
                )
            values.append(value)
        return values


__all__ = [
    "TrainerObjectiveWrapper",
]
