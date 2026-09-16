"""Tests for the callback system: ABC, wrappers, manager, built-in callbacks."""

import gc as gc_module
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from utils.config.run_config import EarlyStoppingConfig
from utils.training.callbacks import (
    Callback,
    CallbackManager,
    CheckpointCallback,
    EarlyStopping,
    EarlyStoppingCallback,
    FunctionCallback,
    MemoryCleanupCallback,
    TestEvaluationCallback,
)
from utils.training.checkpoint import CheckpointManager


class _StubTrainer:
    """Duck-typed trainer exposing exactly the attrs callbacks read."""

    # Set only by tests that fake the test-evaluation entry point.
    _evaluate_on_test_set: Callable[..., None]

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        opt: torch.optim.Optimizer | None = None,
        lr_scheduler: Any = None,
        resumed: bool = False,
        metric_logger: Any = None,
    ) -> None:
        self.model = model or torch.nn.Linear(2, 1)
        self.opt = opt
        self.lr_scheduler = lr_scheduler
        self._resumed = resumed
        self.metric_logger = metric_logger
        self._global_step = 7
        self.calls: list[object] = []


# --- Callback ABC ---


class TestCallbackABC:
    def test_all_hooks_default_noop_accepting_kwargs(self) -> None:
        cb = Callback()
        # getattr keeps the runtime `is None` noop check; a direct comparison
        # against a statically None-returning call is a mypy func-returns-value
        # error.
        assert getattr(cb, "on_train_begin")(3, trainer=None) is None
        assert getattr(cb, "on_train_end")(trainer=None) is None
        assert getattr(cb, "on_epoch_begin")(0, trainer=None) is None
        assert getattr(cb, "on_epoch_end")(0, 0.1, 0.2, trainer=None) is None
        assert getattr(cb, "on_phase_begin")(0, "train", trainer=None) is None
        assert getattr(cb, "on_phase_end")(0, "train", 0.1, {}, trainer=None) is None
        assert getattr(cb, "on_batch_begin")(0, 1, "train", trainer=None) is None
        assert getattr(cb, "on_batch_end")(0, 1, "train", 0.1, trainer=None) is None
        assert cb.should_stop(trainer=None) is False


# --- FunctionCallback ---


class TestFunctionCallback:
    def test_none_handlers_skipped(self) -> None:
        calls: list[int] = []
        # None handlers are deliberately passed (and filtered) at runtime.
        fc = FunctionCallback(
            cast(
                "dict[str, Callable | list[Callable]]",
                {"on_epoch_begin": None, "on_train_end": lambda: calls.append(1)},
            )
        )
        fc.on_epoch_begin(0)
        assert calls == []
        fc.on_train_end()
        assert calls == [1]

    def test_single_handler_normalized_to_list(self) -> None:
        calls: list[object] = []
        fc = FunctionCallback(
            {"on_epoch_end": lambda e, tl, vl: calls.append((e, tl, vl))}
        )
        fc.on_epoch_end(2, 0.5, 0.6)
        assert calls == [(2, 0.5, 0.6)]

    def test_multiple_handlers_called_in_order(self) -> None:
        calls: list[str] = []
        fc = FunctionCallback(
            {
                "on_batch_end": [
                    lambda *a: calls.append("first"),
                    lambda *a: calls.append("second"),
                ]
            }
        )
        fc.on_batch_end(0, 1, "train", 0.1)
        assert calls == ["first", "second"]

    def test_should_stop_any_truthy(self) -> None:
        fc = FunctionCallback({"should_stop": [lambda: False, lambda: True]})
        assert fc.should_stop() is True
        fc2 = FunctionCallback({"should_stop": [lambda: False]})
        assert fc2.should_stop() is False

    def test_trainer_kwarg_not_forwarded(self) -> None:
        # kwargs like trainer= are manager plumbing, not handler arguments.
        seen: dict[str, object] = {}

        def handler(epoch: int) -> None:
            seen["epoch"] = epoch

        fc = FunctionCallback({"on_epoch_begin": handler})
        fc.on_epoch_begin(4, trainer=object())
        assert seen == {"epoch": 4}


# --- CallbackManager ---


class TestCallbackManager:
    def test_none_entries_filtered(self) -> None:
        cb = Callback()
        mgr = CallbackManager(cast("list[Callback]", [None, cb, None]))
        assert mgr.callbacks == [cb]

    def test_trigger_reaches_callbacks_via_getattr(self) -> None:
        class Recorder(Callback):
            def __init__(self) -> None:
                self.events: list[tuple[Any, ...]] = []

            def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
                self.events.append(("epoch_begin", epoch))

        rec = Recorder()
        mgr = CallbackManager([rec])
        mgr.trigger("on_epoch_begin", 5)
        assert rec.events == [("epoch_begin", 5)]

    def test_get_callback_first_isinstance_match(self) -> None:
        first = EarlyStoppingCallback(
            early_stopping=EarlyStopping(EarlyStoppingConfig())
        )
        second = CheckpointCallback(CheckpointManager("."), early_stopping=None)
        mgr = CallbackManager([first, second])
        assert mgr.get_callback(CheckpointCallback) is second
        assert mgr.get_callback(EarlyStoppingCallback) is first

    def test_get_callback_no_match_returns_none(self) -> None:
        mgr = CallbackManager([Callback()])
        assert mgr.get_callback(MemoryCleanupCallback) is None

    def test_should_stop_any(self) -> None:
        class Stopper(Callback):
            def __init__(self, stop: bool) -> None:
                self._stop = stop

            def should_stop(self, **kwargs: Any) -> bool:
                return self._stop

        assert CallbackManager([Stopper(False), Stopper(True)]).should_stop() is True
        assert CallbackManager([Stopper(False)]).should_stop() is False

    def test_close_isolates_callback_failures(self) -> None:
        class Exploding(Callback):
            def close(self) -> None:
                raise RuntimeError("boom")

        class Recording(Callback):
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        recording = Recording()
        # close() runs on the finally path: a raising callback must not
        # propagate (masking the original training error) nor skip the rest.
        CallbackManager([Exploding(), recording]).close()
        assert recording.closed is True


# --- EarlyStoppingCallback ---


class TestEarlyStoppingCallback:
    def _cb(self, **cfg: Any) -> tuple[EarlyStoppingCallback, EarlyStopping]:
        es = EarlyStopping(EarlyStoppingConfig(**cfg))
        return EarlyStoppingCallback(early_stopping=es), es

    def test_train_phase_ignored(self) -> None:
        cb, es = self._cb(patience=1)
        cb.on_phase_end(0, "train", 0.5, {"auc": 0.9})
        assert es.best_score is None

    def test_monitor_loss_reads_val_loss(self) -> None:
        cb, es = self._cb(monitor="loss", mode="min")
        cb.on_phase_end(0, "val", 0.25, {"auc": 0.9})
        assert es.best_score == 0.25

    def test_named_metric_wins_over_loss(self) -> None:
        cb, es = self._cb(monitor="auc")
        cb.on_phase_end(0, "val", 0.25, {"auc": 0.9})
        assert es.best_score == 0.9

    def test_fallback_chain_auc_auprc_acc_rmse(self) -> None:
        cb, es = self._cb(monitor="auc")
        cb._select_monitor_value({}, None)  # nothing available
        cb.on_phase_end(0, "val", cast("float", None), {"rmse": 0.3})
        assert es.best_score == 0.3  # rmse is the last fallback

        cb2, es2 = self._cb(monitor="auc")
        cb2.on_phase_end(
            0, "val", cast("float", None), {"auprc": 0.4, "acc": 0.5, "rmse": 0.6}
        )
        assert es2.best_score == 0.4  # auprc beats acc/rmse in the chain

    def test_missing_monitor_defaults_to_minus_inf_for_max(self) -> None:
        cb, _ = self._cb(monitor="auc")
        assert cb._select_monitor_value({}, None) == float("-inf")

    def test_missing_monitor_defaults_to_plus_inf_for_min_names(self) -> None:
        cb, _ = self._cb(monitor="rmse", mode="min")
        assert cb._select_monitor_value({}, None) == float("inf")

    def test_should_stop_delegates_to_cached_stop(self) -> None:
        cb, _ = self._cb(patience=1)
        assert cb.should_stop() is False
        cb.on_phase_end(0, "val", 0.5, {"auc": 0.5})
        cb.on_phase_end(1, "val", 0.4, {"auc": 0.4})  # bad epoch -> stop
        assert cb.should_stop() is True

    def test_metric_logger_notified_when_trainer_has_one(self) -> None:
        class _RecordingLogger:
            def __init__(self) -> None:
                self.events: list[dict[str, Any]] = []

            def log_early_stopping(self, **kwargs: Any) -> None:
                self.events.append(kwargs)

        rec = _RecordingLogger()
        trainer = _StubTrainer(metric_logger=rec)
        cb, _ = self._cb()
        cb.on_phase_end(0, "val", 0.3, {"auc": 0.8}, trainer=trainer)
        assert len(rec.events) == 1
        event = rec.events[0]
        assert event["best_score"] == 0.8
        assert event["epoch"] == 0
        assert event["step"] == trainer._global_step

    def test_no_trainer_no_logger_crash(self) -> None:
        cb, _ = self._cb()
        cb.on_phase_end(0, "val", 0.3, {"auc": 0.8})  # trainer kwarg absent


# --- CheckpointCallback ---


class TestCheckpointCallback:
    def _mgr(self, tmp_path: Path) -> CheckpointManager:
        return CheckpointManager(str(tmp_path))

    def test_phase_end_requires_trainer_val_and_best_filename(
        self, tmp_path: Path
    ) -> None:
        mgr = self._mgr(tmp_path)
        model = torch.nn.Linear(2, 1)
        cb = CheckpointCallback(mgr, early_stopping=None, best_filename="best.pth")
        cb.on_phase_end(0, "val", 0.1, {"auc": 0.9})  # no trainer
        cb.on_phase_end(0, "train", 0.1, {"auc": 0.9}, trainer=_StubTrainer(model))
        assert not (tmp_path / "best.pth").exists()
        cb.best_filename = None
        cb.on_phase_end(0, "val", 0.1, {"auc": 0.9}, trainer=_StubTrainer(model))
        assert not (tmp_path / "best.pth").exists()

    def test_saves_best_and_caches_state(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        trainer = _StubTrainer()
        cb = CheckpointCallback(mgr, early_stopping=None)
        cb.on_phase_end(0, "val", 0.1, {"auc": 0.9}, trainer=trainer)
        cb.on_phase_end(1, "val", 0.1, {"auc": 0.7}, trainer=trainer)  # worse
        mgr.flush()  # saves are async; drain before asserting on disk
        assert (tmp_path / "best_model.pth").exists()
        assert cb.best_metric == 0.9
        assert cb.best_epoch == 0
        assert isinstance(cb.best_model_state, dict)

    def test_keep_best_state_false_no_cache(self, tmp_path: Path) -> None:
        mgr = self._mgr(tmp_path)
        cb = CheckpointCallback(mgr, early_stopping=None, keep_best_state=False)
        cb.on_phase_end(0, "val", 0.1, {"auc": 0.9}, trainer=_StubTrainer())
        assert cb.best_model_state is None
        mgr.flush()  # saves are async; drain before asserting on disk
        assert (tmp_path / "best_model.pth").exists()

    def test_is_better_first_always_wins(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(self._mgr(tmp_path), early_stopping=None)
        assert cb._is_better_metric(float("-inf")) is True

    def test_mode_precedence_override_beats_es(self, tmp_path: Path) -> None:
        es = EarlyStopping(EarlyStoppingConfig(monitor="auc", mode="max"))
        cb = CheckpointCallback(self._mgr(tmp_path), early_stopping=es, mode="min")
        cb.best_metric = 0.5
        assert cb._is_better_metric(0.4) is True  # min override wins
        assert cb._is_better_metric(0.6) is False

    def test_mode_falls_back_to_es_cfg(self, tmp_path: Path) -> None:
        es = EarlyStopping(EarlyStoppingConfig(monitor="rmse", mode="min"))
        cb = CheckpointCallback(self._mgr(tmp_path), early_stopping=es)
        cb.best_metric = 0.5
        assert cb._is_better_metric(0.4) is True

    def test_mode_name_convention_without_es(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(
            self._mgr(tmp_path), early_stopping=None, monitor="rmse"
        )
        cb.best_metric = 0.5
        assert cb._is_better_metric(0.4) is True
        assert cb._is_better_metric(0.6) is False

    def test_equal_value_not_better_strict_compare(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(self._mgr(tmp_path), early_stopping=None)
        cb.best_metric = 0.5
        assert cb._is_better_metric(0.5) is False

    def test_on_train_begin_resets_unless_resumed(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(self._mgr(tmp_path), early_stopping=None)
        cb.best_metric = 0.9
        cb.best_epoch = 3
        cb.best_model_state = {"w": 1}
        cb.on_train_begin(5, trainer=_StubTrainer())
        assert cb.best_metric is None and cb.best_epoch is None
        assert cb.best_model_state is None

        cb.best_metric = 0.9
        cb.on_train_begin(5, trainer=_StubTrainer(resumed=True))
        assert cb.best_metric == 0.9  # resumed run keeps its tracking

    def test_epoch_end_saves_last_with_es_state(self, tmp_path: Path) -> None:
        es = EarlyStopping(EarlyStoppingConfig())
        es.step(0.8, epoch=1, metrics={"auc": 0.8})
        cb = CheckpointCallback(self._mgr(tmp_path), early_stopping=es)
        trainer = _StubTrainer()
        trainer.opt = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        cb.on_epoch_end(1, 0.5, 0.4, trainer=trainer)
        cb.checkpoint_manager.flush()
        saved = torch.load(tmp_path / "last_checkpoint.pth", weights_only=False)
        assert saved["early_stopping_state"]["best_score"] == 0.8
        assert saved["epoch"] == 1

    def test_epoch_end_skipped_when_disabled(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(
            self._mgr(tmp_path), early_stopping=None, save_last_checkpoint=False
        )
        cb.on_epoch_end(1, 0.5, 0.4, trainer=_StubTrainer())
        assert not (tmp_path / "last_checkpoint.pth").exists()


# --- MemoryCleanupCallback ---


class TestMemoryCleanupCallback:
    def test_fires_on_interval_multiple_including_epoch_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fired: list[str] = []
        cb = MemoryCleanupCallback(cleanup_interval=3)
        monkeypatch.setattr(cb, "_cleanup_memory", lambda phase: fired.append(phase))
        for epoch in range(7):
            cb.on_phase_end(epoch, "train", 0.1, {})
        assert fired == ["train", "train", "train"]  # epochs 0, 3, 6

    def test_always_gcs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        count = {"n": 0}
        real_collect = gc_module.collect

        def counting_collect() -> int:
            count["n"] += 1
            return real_collect()

        monkeypatch.setattr(gc_module, "collect", counting_collect)
        cb = MemoryCleanupCallback(cleanup_interval=1)
        cb.on_phase_end(0, "train", 0.1, {})
        assert count["n"] == 1


# --- TestEvaluationCallback ---


class TestTestEvaluationCallback:
    def test_delegates_to_trainer_on_train_end(self) -> None:
        trainer = _StubTrainer()
        trainer._evaluate_on_test_set = lambda use_best_model: trainer.calls.append(
            ("eval", use_best_model)
        )
        cb = TestEvaluationCallback(use_best_model=False)
        cb.on_train_end(trainer=trainer)
        assert trainer.calls == [("eval", False)]

    def test_no_trainer_is_noop(self) -> None:
        TestEvaluationCallback().on_train_end()  # must not raise
