"""Optuna tuner wrapper and utility tools."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna
import yaml
from optuna.samplers import GridSampler

from utils.core import get_logger

if TYPE_CHECKING:
    import pandas as pd

from .config import HyperparameterSpace, OptunaConfig

logger = get_logger(__name__)


class OptunaTuner:
    """Optuna hyperparameter search engine."""

    def __init__(
        self,
        config: OptunaConfig,
        param_space: list[HyperparameterSpace],
        objective_fn: Callable[..., float | list[float]],
        objective_kwargs: dict[str, Any] | None = None,
    ):
        """Initialise the Optuna tuner.

        Args:
            config: Optuna search configuration.
            param_space: List of hyperparameter space definitions.
            objective_fn: Objective function to optimise.
            objective_kwargs: Extra keyword arguments for the objective function.
        """
        self.config = config
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.objective_kwargs = objective_kwargs or {}

        # Validate parameter space
        for space in self.param_space:
            space.validate()

        # Create study
        self.study: optuna.Study | None = None
        self._param_importances: dict[str, float] | None = None
        self._pareto_front: list[optuna.trial.FrozenTrial] | None = None
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure Optuna logging verbosity."""
        if self.config.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.config.verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

    def _objective(self, trial: optuna.trial.Trial) -> float | list[float]:
        """Optuna objective function wrapper.

        Samples hyperparameters from the search space and invokes the
        user-defined objective function.

        Args:
            trial: Optuna trial object.

        Returns:
            The objective score for this trial.
        """
        # Sample hyperparameters from the search space
        params = {}
        for space in self.param_space:
            params[space.name] = space.suggest(trial)

        # Call user-defined objective function
        score = self.objective_fn(trial, params=params, **self.objective_kwargs)

        return score

    def search(self) -> dict[str, Any]:
        """Execute the hyperparameter search.

        Returns:
            Dictionary of the best found parameters.
        """
        # Create study
        sampler = self.config.get_sampler()
        pruner = self.config.get_pruner()

        storage_url = None
        if self.config.db_url:
            storage_url = self.config.db_url
        elif self.config.save_dir:
            os.makedirs(self.config.save_dir, exist_ok=True)
            storage_url = f"sqlite:///{os.path.join(self.config.save_dir, 'study.db')}"

        study_name = (
            self.config.study_name
            or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        # Persist so --resume reuses the name; otherwise a fresh timestamp
        # creates an empty study and the prior trials are lost.
        self.config.study_name = study_name

        study_kwargs: dict[str, Any] = {
            "sampler": sampler,
            "pruner": pruner,
            "study_name": study_name,
            "storage": storage_url,
            "load_if_exists": True,
        }

        directions = self.config.directions
        if isinstance(directions, list):
            cleaned = [d for d in directions if d]
            if len(cleaned) == 1:
                study_kwargs["direction"] = cleaned[0]
            elif len(cleaned) > 1:
                study_kwargs["directions"] = cleaned
            else:
                raise ValueError("Optuna directions list is empty")
        elif directions:
            study_kwargs["direction"] = directions
        else:
            raise ValueError("Optuna direction configuration missing")

        self.study = optuna.create_study(**study_kwargs)

        # GridSampler exhausts all combinations; no default seed needed
        if not isinstance(sampler, GridSampler):
            self._enqueue_defaults()

        if self.config.n_jobs > 1:
            logger.warning(
                "n_jobs>1 runs trials in threads sharing one GPU/process, which "
                "can cause GPU memory/compute contention. Recommended only for "
                "CPU models or a dedicated multi-GPU setup."
            )

        # Optimise; failed trials are marked FAIL without stopping the search
        self.study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            catch=(Exception,),
            show_progress_bar=(self.config.verbose > 0),
        )

        # Assess fANOVA parameter importances (best-effort)
        self._param_importances = self._compute_param_importances()

        # Resolve the Pareto front once; reused by _best_trial / _best_params /
        # print_summary so the non-dominated sort is not repeated each call.
        if len(self.study.directions) > 1:
            self._pareto_front = self.study.best_trials

        # Save results
        if self.config.save_dir:
            self._save_results()

        # Surface a clear error if nothing completed; done after _save_results so
        # failure details (recorded via trial user_attrs) are still persisted.
        self._raise_if_all_failed()

        return self._best_params()

    def _enqueue_defaults(self) -> None:
        """Enqueue declared defaults as a startup trial.

        Speeds up sampler convergence. Parameters without declared defaults
        are sampled normally during the trial. Skipped on resume so the
        enqueued trial does not eat into ``n_trials`` a second time.
        """
        assert self.study is not None, "search() must create the study first"
        if self.study.trials:
            return
        defaults = {
            space.name: space.default
            for space in self.param_space
            if space.default is not None
        }
        if not defaults:
            return
        self.study.enqueue_trial(defaults)
        logger.info(f"Enqueued {len(defaults)} default params as a startup trial")

    def _best_params(self) -> dict[str, Any]:
        """Return the best parameters.

        Delegates to :meth:`_best_trial`, so single- and multi-objective share a
        single selection path.

        Returns:
            Dictionary of best parameters, or empty dict if unavailable.
        """
        best = self._best_trial()
        return best.params if best is not None else {}

    def _best_trial(self) -> optuna.trial.FrozenTrial | None:
        """Return the best trial, or None if unavailable.

        Single-objective returns ``study.best_trial``. Multi-objective picks one
        trial from the Pareto front via :meth:`_pick_pareto_representative` (a
        deterministic tie-break) so the choice is stable across resumes. The
        result is always a COMPLETE trial.
        """
        assert self.study is not None, "search() must run first"
        if len(self.study.directions) > 1:
            pareto = self._pareto_front
            if pareto is None:
                pareto = self.study.best_trials
            if not pareto:
                return None
            return self._pick_pareto_representative(pareto)
        try:
            return self.study.best_trial
        except ValueError:
            logger.warning("No completed trials; cannot determine best trial")
            return None

    def _pick_pareto_representative(
        self, pareto: list[optuna.trial.FrozenTrial]
    ) -> optuna.trial.FrozenTrial:
        """Deterministically pick one trial from a Pareto front.

        Front members are all non-dominated, so the pick is conventional: best on
        the first objective (in its optimisation direction), ties broken by
        lowest trial number. This makes the representative stable and
        reproducible across resumes and optuna versions.
        """
        assert self.study is not None, "search() must run first"
        first_dir = self.study.directions[0]
        if first_dir == optuna.study.StudyDirection.MAXIMIZE:
            return min(pareto, key=lambda t: (-t.values[0], t.number))
        return min(pareto, key=lambda t: (t.values[0], t.number))

    def _raise_if_all_failed(self) -> None:
        """Log trial-outcome counts and raise only when trials actually failed.

        On total failure, attach the last recorded traceback (stored on the trial
        by ``TrainerObjectiveWrapper``) so the root cause surfaces instead of a
        silent empty ``best_params``. An all-pruned run is not a failure.
        """
        assert self.study is not None, "search() must run first"
        counts = Counter(t.state for t in self.study.trials)
        n_complete = counts.get(optuna.trial.TrialState.COMPLETE, 0)
        n_fail = counts.get(optuna.trial.TrialState.FAIL, 0)
        n_pruned = counts.get(optuna.trial.TrialState.PRUNED, 0)
        # Enqueued defaults sit in WAITING; exclude them from the evaluated total.
        n_total = sum(
            1 for t in self.study.trials if t.state != optuna.trial.TrialState.WAITING
        )
        logger.info(
            f"Trial outcomes: {n_complete} COMPLETE, {n_fail} FAIL, {n_pruned} PRUNED"
        )
        if n_complete > 0:
            return
        if n_fail == 0:
            logger.warning(
                f"No trial completed out of {n_total} (0 COMPLETE, 0 FAIL, "
                f"{n_pruned} PRUNED); all evaluated trials were pruned."
            )
            return

        detail = ""
        failed = next(
            (t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL),
            None,
        )
        if failed is not None:
            err = failed.user_attrs.get("error", "unknown")
            tb = failed.user_attrs.get("traceback")
            detail = (
                f"\nLast failure ({err}):\n{tb}" if tb else f"\nLast failure: {err}"
            )
        raise RuntimeError(
            f"All {n_total} trial(s) failed "
            f"(0 COMPLETE, {n_fail} FAIL, {n_pruned} PRUNED).{detail}"
        )

    def _save_best_run_config(self) -> None:
        """Copy the best trial's full ``run_config.yaml`` for reproduction.

        Each trial archives its complete RunConfig under ``trial_<N>/``. Copying
        the best trial's archive to ``best_run_config.yaml`` lets the result be
        reproduced with ``python train.py --config best_run_config.yaml``.
        """
        assert self.study is not None, "search() must run first"
        assert self.config.save_dir is not None, "save_dir must be configured"
        best = self._best_trial()
        if best is None:
            logger.warning("No best trial; skipping best_run_config.yaml")
            return

        trial_dir = best.user_attrs.get("trial_dir")
        if trial_dir is None:
            trial_dir = os.path.join(self.config.save_dir, f"trial_{best.number}")
        src = os.path.join(trial_dir, "run_config.yaml")
        if not os.path.isfile(src):
            logger.warning(
                f"Best trial run_config not found at {src}; "
                "skipping best_run_config.yaml"
            )
            return

        dst = os.path.join(self.config.save_dir, "best_run_config.yaml")
        shutil.copy2(src, dst)
        # Multi-objective: this is one representative of the Pareto front, not a
        # unique optimum (see _pick_pareto_representative).
        is_multi = len(self.study.directions) > 1
        note = " (representative of the Pareto front)" if is_multi else ""
        logger.info(
            f"Best run config saved to {dst}{note}\n"
            f"Reproduce with: python train.py --config {dst}"
        )

    def _compute_param_importances(self) -> dict[str, float] | None:
        """Assess fANOVA parameter importances from completed trials.

        Best-effort: returns None when too few trials completed or the evaluator
        errors, so importance never blocks result saving. Multi-objective studies
        assess importance against the first objective.
        """
        from optuna.importance import FanovaImportanceEvaluator, get_param_importances

        assert self.study is not None, "search() must run first"
        completed = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < 4:
            logger.info(
                f"Skipping param importance: {len(completed)} completed trial(s) "
                "(need >= 4 for a stable fANOVA estimate)"
            )
            return None

        evaluator = FanovaImportanceEvaluator(seed=self.config.seed)

        def target(trial: optuna.trial.FrozenTrial) -> float:
            return trial.values[0]

        try:
            importances = get_param_importances(
                self.study,
                evaluator=evaluator,
                target=target if len(self.study.directions) > 1 else None,
            )
        except Exception as e:
            logger.warning(f"Param importance evaluation failed: {e}")
            return None
        # Optuna's fANOVA evaluator returns numpy float64 values, which yaml
        # cannot serialize; coerce to plain floats.
        return {name: float(value) for name, value in importances.items()}

    def _save_results(self) -> None:
        """Save search results to disk as yaml."""
        if not self.study or not self.config.save_dir:
            return

        os.makedirs(self.config.save_dir, exist_ok=True)

        # Best parameters
        best_params_path = os.path.join(self.config.save_dir, "best_params.yaml")
        Path(best_params_path).write_text(
            yaml.safe_dump(self._best_params(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Best trial's full run_config.yaml for one-command reproduction
        try:
            self._save_best_run_config()
        except Exception as e:
            logger.warning(f"Failed to save best_run_config.yaml: {e}")

        # Search history
        history_path = os.path.join(self.config.save_dir, "search_history.yaml")
        multi_objective = len(self.study.directions) > 1
        trials_data = [
            {
                "number": trial.number,
                # FrozenTrial.value RAISES on multi-objective studies (optuna
                # >= 4), so branch on the study shape, not on the value.
                "value": trial.values if multi_objective else trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "error": trial.user_attrs.get("error"),
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "duration_sec": trial.user_attrs.get("duration_sec"),
            }
            for trial in self.study.trials
        ]
        Path(history_path).write_text(
            yaml.safe_dump(trials_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Parameter importances (fANOVA)
        if self._param_importances:
            importance_path = os.path.join(
                self.config.save_dir, "param_importances.yaml"
            )
            Path(importance_path).write_text(
                yaml.safe_dump(
                    self._param_importances, sort_keys=False, allow_unicode=True
                ),
                encoding="utf-8",
            )

        # Configuration echo
        config_path = os.path.join(self.config.save_dir, "optuna_config.yaml")
        Path(config_path).write_text(
            yaml.safe_dump(
                asdict(self.config) if is_dataclass(self.config) else self.config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        logger.info(f"Results saved to {self.config.save_dir}")

    def print_summary(self) -> None:
        """Print a summary of the search results."""
        if not self.study:
            logger.warning("No study found. Run search() first.")
            return

        state_counts = Counter(t.state.name for t in self.study.trials)
        log = [
            "=" * 60,
            "Optuna Hyperparameter Search Summary",
            "=" * 60,
            f"Study Name: {self.study.study_name}",
            f"Total Trials: {len(self.study.trials)}",
            (
                f"COMPLETE: {state_counts.get('COMPLETE', 0)}, "
                f"FAIL: {state_counts.get('FAIL', 0)}, "
                f"PRUNED: {state_counts.get('PRUNED', 0)}"
            ),
        ]
        if len(self.study.directions) > 1:
            pareto = self._pareto_front or self.study.best_trials
            log.append(f"Pareto-optimal trials: {len(pareto)}")
            for t in pareto[:5]:
                log.append(f"  trial {t.number}: values={t.values}")
        else:
            try:
                log.append(f"Best Value: {self.study.best_value}")
                log.append("\nBest Parameters:")
                for param, value in self.study.best_params.items():
                    log.append(f"  {param}: {value}")
            except ValueError:
                log.append("No completed trials.")

        if self._param_importances:
            log.append("\nParameter Importances (fANOVA, top 5):")
            for param, imp in list(self._param_importances.items())[:5]:
                log.append(f"  {param}: {imp:.4f}")

        logger.info("\n".join(log))

    def get_dataframe(self) -> pd.DataFrame | None:
        """Get the trials dataframe (requires pandas).

        Returns:
            DataFrame of trial results, or None if unavailable.
        """
        if not self.study:
            return None
        try:
            return self.study.trials_dataframe()
        except Exception as e:
            logger.error(f"Failed to get dataframe: {e}")
            return None


__all__ = [
    "OptunaTuner",
]
