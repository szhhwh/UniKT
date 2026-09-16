"""Tests for TrainerObjectiveWrapper: rc isolation, metric extraction, error paths.

The trainer is a duck-typed double: the wrapper only touches ``add_callback``,
``run``, ``release``, and the ``early_stopping`` attribute after construction.
"""

from collections.abc import Callable
from typing import Any, cast

import optuna
import pytest

from utils.config.run_config import RunConfig
from utils.optuna_utils.callback import MultiMetricTracker, OptunaTrialCallback
from utils.optuna_utils.trainer import TrainerObjectiveWrapper


class _FakeEarlyStopping:
    def __init__(
        self,
        best_metrics: dict[str, float] | None = None,
        best_score: float | None = None,
        best_epoch: int | None = None,
    ) -> None:
        self.best_metrics = best_metrics or {}
        self.best_score = best_score
        self.best_epoch = best_epoch


class _FakeTrainer:
    """Records callbacks; optional early_stopping attrs; optional failure."""

    # rc is attached post-construction by the factory (mirrors the real
    # BaseTrainer surface the wrapper reads back).
    rc: Any

    def __init__(
        self,
        early_stopping: _FakeEarlyStopping | None = None,
        run_error: Exception | None = None,
        release_error: Exception | None = None,
    ) -> None:
        self.callbacks: list[Any] = []
        self.early_stopping = early_stopping
        self._run_error = run_error
        self._release_error = release_error
        self.released = False

    def add_callback(self, cb: Any) -> None:
        self.callbacks.append(cb)

    def run(self) -> None:
        if self._run_error is not None:
            raise self._run_error

    def release(self) -> None:
        self.released = True
        if self._release_error is not None:
            raise self._release_error


class _TrainerFactory:
    """Trainer-class double: builds a fake trainer, counts constructions."""

    instances: list[_FakeTrainer]

    def __init__(self, **trainer_kwargs: Any) -> None:
        self._kwargs = trainer_kwargs
        self.instances = []

    def __call__(self, *, rc: Any, data_src: Any, exp_manager: Any) -> _FakeTrainer:
        trainer = _FakeTrainer(**self._kwargs)
        trainer.rc = rc
        self.instances.append(trainer)
        return trainer


def _make_wrapper(
    base_rc: Any, trainer_factory: Any, metric: str = "auc"
) -> TrainerObjectiveWrapper:
    # The factory double is a callable instance, not a real ``type``; the
    # wrapper's ``trainer_class`` annotation is narrower than its duck-typed
    # contract, so the boundary crossing is cast explicitly.
    return TrainerObjectiveWrapper(
        trainer_class=cast(type, trainer_factory),
        data_src_fn=lambda: None,
        base_rc=base_rc,
        metric_name=metric,
    )


class _FakePruningCallback:
    def __init__(self, best_value: float | None = None, pruned: bool = False) -> None:
        self.best_value = best_value
        self.pruned = pruned


# --- _create_trial_rc ---


class TestCreateTrialRc:
    def test_params_applied_to_model_node(
        self, make_run_config: Callable[..., RunConfig], registry_snapshot: None
    ) -> None:
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trial_rc = wrapper._create_trial_rc({"hidden_dim": 64})
        assert trial_rc.model.hidden_dim == 64

    def test_base_rc_untouched_by_trial_mutation(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        # hidden_dim lives on the registered TinyTestModelConfig subclass, not
        # on the ModelConfig contract, so the local stays Any.
        base: Any = make_run_config()
        wrapper = _make_wrapper(base, _TrainerFactory())
        trial_rc = wrapper._create_trial_rc({"hidden_dim": 64})
        trial_rc.model.hidden_dim = 128
        trial_rc.general.seed = 0
        assert base.model.hidden_dim == 8
        assert base.general.seed == 42

    def test_trials_are_independent_copies(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        base: Any = make_run_config()
        wrapper = _make_wrapper(base, _TrainerFactory())
        first = wrapper._create_trial_rc({"hidden_dim": 64})
        second = wrapper._create_trial_rc({"hidden_dim": 16})
        assert first.model.hidden_dim == 64
        assert second.model.hidden_dim == 16
        assert first is not second

    def test_auto_pin_memory_disabled_for_trials(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        assert wrapper._create_trial_rc({}).general.pin_memory is False

    def test_explicit_pin_memory_setting_respected(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = _make_wrapper(make_run_config(pin_memory=True), _TrainerFactory())
        assert wrapper._create_trial_rc({}).general.pin_memory is True


# --- _extract_metric (single-objective) ---


class TestExtractMetric:
    def test_pruning_callback_best_value_wins(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(
            early_stopping=_FakeEarlyStopping(best_metrics={"auc": 0.5})
        )
        value = wrapper._extract_metric(
            trainer,
            cast(OptunaTrialCallback | None, _FakePruningCallback(best_value=0.9)),
        )
        assert value == 0.9

    def test_early_stopping_metrics_used_as_fallback(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(
            early_stopping=_FakeEarlyStopping(best_metrics={"auc": 0.8})
        )
        value = wrapper._extract_metric(
            trainer,
            cast(OptunaTrialCallback | None, _FakePruningCallback(best_value=None)),
        )
        assert value == 0.8

    def test_refuses_early_stopping_best_score_surrogate(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(
            early_stopping=_FakeEarlyStopping(best_metrics={}, best_score=0.9)
        )
        with pytest.raises(RuntimeError, match="Could not extract metric 'auc'"):
            wrapper._extract_metric(
                trainer,
                cast(OptunaTrialCallback | None, _FakePruningCallback(best_value=None)),
            )

    def test_no_trainer_state_at_all_raises(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(early_stopping=None)
        with pytest.raises(RuntimeError, match="Could not extract metric 'auc'"):
            wrapper._extract_metric(trainer, None)


# --- _extract_multi (multi-objective) ---


class _StubTracker:
    """Duck-typed MultiMetricTracker: a plain best_values mapping."""

    def __init__(self, best_values: dict[str, float | None]) -> None:
        self.best_values = best_values


class TestExtractMulti:
    def test_missing_tracker_raises(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = TrainerObjectiveWrapper(
            trainer_class=cast(type, _TrainerFactory()),
            data_src_fn=lambda: None,
            base_rc=make_run_config(),
            metric_name=["auc", "rmse"],
        )
        with pytest.raises(RuntimeError, match="tracker was not registered"):
            wrapper._extract_multi(None)

    def test_missing_metric_raises_with_tracked_summary(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = TrainerObjectiveWrapper(
            trainer_class=cast(type, _TrainerFactory()),
            data_src_fn=lambda: None,
            base_rc=make_run_config(),
            metric_name=["auc", "rmse"],
        )
        tracker = _StubTracker({"auc": 0.7, "rmse": None})
        with pytest.raises(RuntimeError, match="Could not extract metric 'rmse'"):
            wrapper._extract_multi(cast(MultiMetricTracker | None, tracker))

    def test_returns_per_objective_values(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        wrapper = TrainerObjectiveWrapper(
            trainer_class=cast(type, _TrainerFactory()),
            data_src_fn=lambda: None,
            base_rc=make_run_config(),
            metric_name=["auc", "rmse"],
        )
        assert wrapper._extract_multi(
            cast(MultiMetricTracker | None, _StubTracker({"auc": 0.7, "rmse": 0.2}))
        ) == [
            0.7,
            0.2,
        ]


# --- __call__ integration on fake trainers ---


class TestCall:
    def test_happy_path_returns_metric_and_registers_callback(
        self,
        make_run_config: Callable[..., RunConfig],
        fake_trial: Any,
    ) -> None:
        factory = _TrainerFactory(
            early_stopping=_FakeEarlyStopping(best_metrics={"auc": 0.42})
        )
        wrapper = _make_wrapper(make_run_config(), factory)
        value = wrapper(fake_trial, params={"hidden_dim": 64})
        assert value == 0.42
        # one pruning callback registered; params landed on the trial rc
        assert len(factory.instances[0].callbacks) == 1
        assert isinstance(factory.instances[0].callbacks[0].trial, type(fake_trial))
        assert factory.instances[0].rc.model.hidden_dim == 64
        assert fake_trial.user_attrs["best_epoch"] is None
        assert "duration_sec" in fake_trial.user_attrs

    def test_error_recorded_in_user_attrs_and_reraised(
        self,
        make_run_config: Callable[..., RunConfig],
        fake_trial: Any,
    ) -> None:
        factory = _TrainerFactory(run_error=ValueError("bad dataset"))
        wrapper = _make_wrapper(make_run_config(), factory)
        with pytest.raises(ValueError, match="bad dataset"):
            wrapper(fake_trial)
        assert "ValueError('bad dataset')" in fake_trial.user_attrs["error"]
        assert "ValueError" in fake_trial.user_attrs["traceback"]
        assert "duration_sec" in fake_trial.user_attrs

    def test_trial_pruned_reraised_without_error_attr(
        self,
        make_run_config: Callable[..., RunConfig],
        fake_trial: Any,
    ) -> None:
        factory = _TrainerFactory(run_error=optuna.TrialPruned("mid-run"))
        wrapper = _make_wrapper(make_run_config(), factory)
        with pytest.raises(optuna.TrialPruned):
            wrapper(fake_trial)
        assert "error" not in fake_trial.user_attrs


# --- release() on every exit path ---


class TestTrialRelease:
    """The wrapper must release each trial's trainer, whatever the outcome."""

    @pytest.mark.parametrize(
        "run_error",
        [None, ValueError("bad dataset"), optuna.TrialPruned("mid-run")],
        ids=["happy", "error", "pruned"],
    )
    def test_release_called_on_every_exit_path(
        self,
        make_run_config: Callable[..., RunConfig],
        fake_trial: Any,
        run_error: Exception | None,
    ) -> None:
        factory = _TrainerFactory(
            early_stopping=_FakeEarlyStopping(best_metrics={"auc": 0.42}),
            run_error=run_error,
        )
        wrapper = _make_wrapper(make_run_config(), factory)
        if run_error is None:
            wrapper(fake_trial)
        else:
            with pytest.raises(type(run_error)):
                wrapper(fake_trial)
        assert factory.instances[0].released

    def test_release_failure_does_not_mask_original_error(
        self,
        make_run_config: Callable[..., RunConfig],
        fake_trial: Any,
    ) -> None:
        factory = _TrainerFactory(
            run_error=ValueError("bad dataset"), release_error=RuntimeError("late")
        )
        wrapper = _make_wrapper(make_run_config(), factory)
        with pytest.raises(ValueError, match="bad dataset"):
            wrapper(fake_trial)
