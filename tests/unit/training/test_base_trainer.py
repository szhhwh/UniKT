"""Tests for ``utils.training.base_trainer``: build, run lifecycle, epoch loop.

Uses the conftest ``TinyTrainer`` (Linear regression over plain ``(x, y)``
tuple batches) so the full template-method flow runs on CPU without any
DataLoader workers. Filesystem effects stay inside the stub exp manager's
tmp_path log dir.
"""

from __future__ import annotations

import csv
import pathlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pytest
import torch

from tests.unit.training.conftest import StubExpManager, TinyTrainer
from utils.config.run_config import EarlyStoppingConfig
from utils.training import base_trainer as base_trainer_module
from utils.training.base_trainer import StageResult
from utils.training.callbacks import (
    Callback,
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    MemoryCleanupCallback,
    TestEvaluationCallback,
)
from utils.training.checkpoint import CheckpointManager
from utils.training.metric_logger import (
    AsyncMetricLoggerProxy,
    LocalMetricLogger,
    MetricLogger,
    MetricLoggerComposite,
)
from utils.training.runtime_components import RuntimeComponents

if TYPE_CHECKING:
    from utils.config.run_config import RunConfig


class _RecordingCallback(Callback):
    """Records every hook invocation with its positional args."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def on_train_begin(self, epochs: int, **kwargs: Any) -> None:
        self.events.append(("train_begin", epochs))

    def on_train_end(self, **kwargs: Any) -> None:
        self.events.append(("train_end",))

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        self.events.append(("epoch_begin", epoch))

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float | None, **kwargs: Any
    ) -> None:
        self.events.append(("epoch_end", epoch, train_loss, val_loss))

    def on_phase_begin(self, epoch: int, phase: str, **kwargs: Any) -> None:
        self.events.append(("phase_begin", epoch, phase))

    def on_phase_end(
        self,
        epoch: int,
        phase: str,
        loss: float | None,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> None:
        self.events.append(("phase_end", epoch, phase, loss))

    def on_batch_begin(
        self, epoch: int, batch_idx: int, phase: str, **kwargs: Any
    ) -> None:
        self.events.append(("batch_begin", epoch, batch_idx, phase))

    def on_batch_end(
        self,
        epoch: int,
        batch_idx: int,
        phase: str,
        loss: float,
        **kwargs: Any,
    ) -> None:
        self.events.append(("batch_end", epoch, batch_idx, phase, loss))


class _CountingScheduler:
    """Duck-typed lr scheduler counting step() calls."""

    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _RecordingMetricLogger:
    """Duck-typed metric logger recording method names."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def init_run(self, **kwargs: Any) -> None:
        self.calls.append("init_run")

    def log_metrics(self, **kwargs: Any) -> None:
        self.calls.append("log_metrics")

    def log_early_stopping(self, **kwargs: Any) -> None:
        self.calls.append("log_early_stopping")

    def log_batch(self, **kwargs: Any) -> None:
        self.calls.append("log_batch")

    def log_final(self, **kwargs: Any) -> None:
        self.calls.append("log_final")

    def log_timing(self, **kwargs: Any) -> None:
        self.calls.append("log_timing")

    def finish(self) -> None:
        self.calls.append("finish")


def _find_local_backend(logger: object) -> LocalMetricLogger | None:
    """Unwrap Composite/AsyncProxy layers down to the LocalMetricLogger."""
    if isinstance(logger, LocalMetricLogger):
        return logger
    if isinstance(logger, AsyncMetricLoggerProxy):
        return _find_local_backend(logger._inner)
    if isinstance(logger, MetricLoggerComposite):
        for inner in logger._loggers:
            found = _find_local_backend(inner)
            if found is not None:
                return found
    return None


_C = TypeVar("_C", bound=Callback)


def _manager(trainer: TinyTrainer) -> CallbackManager:
    """Callback manager of an already-built trainer (set during build())."""
    return cast(CallbackManager, trainer.callback_manager)


def _required_cb(trainer: TinyTrainer, cb_type: type[_C]) -> _C:
    """A callback build() is known to have registered; never None post-build."""
    return cast(_C, _manager(trainer).get_callback(cb_type))


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


class TestBuild:
    def test_rc_none_raises_value_error(
        self,
        make_exp_manager: Callable[..., StubExpManager],
        make_batches: Callable[..., list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        with pytest.raises(ValueError, match="run_config is required"):
            TinyTrainer(None, make_exp_manager(), make_batches())

    def test_exp_manager_none_raises_value_error(
        self,
        make_run_config: Callable[..., RunConfig],
        make_batches: Callable[..., list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        with pytest.raises(ValueError, match="exp_manager is required"):
            TinyTrainer(make_run_config(), None, make_batches())

    def test_build_sets_state(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer(model_kwargs={"epochs": 3})
        assert trainer._built is True
        assert trainer.device_ == torch.device("cpu")
        assert trainer.epochs == 3  # per-run snapshot read from rc.model
        assert trainer.log_dir == trainer._exp_manager.get_log_dir()
        assert isinstance(trainer._components, RuntimeComponents)

    def test_builtin_callback_composition_skip_test(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()  # make_run_config default: skip_test=True
        kinds = {type(cb) for cb in _manager(trainer).callbacks}
        assert {
            MemoryCleanupCallback,
            EarlyStoppingCallback,
            CheckpointCallback,
        } <= kinds
        assert TestEvaluationCallback not in kinds

    def test_test_evaluation_callback_added_when_not_skipping(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
        make_batches: Callable[..., list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        rc = make_run_config(skip_test=False, model_kwargs={"epochs": 1})
        trainer = make_tiny_trainer(rc=rc, test=make_batches())
        kinds = {type(cb) for cb in _manager(trainer).callbacks}
        assert TestEvaluationCallback in kinds

    def test_early_stopping_always_constructed(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()
        assert isinstance(trainer.early_stopping, base_trainer_module.EarlyStopping)
        es_cb = _required_cb(trainer, EarlyStoppingCallback)
        assert es_cb.early_stopping is trainer.early_stopping

    def test_run_config_archived_unless_existing_run(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_exp_manager: Callable[..., StubExpManager],
    ) -> None:
        fresh = make_tiny_trainer()
        assert (pathlib.Path(cast(str, fresh.log_dir)) / "run_config.yaml").exists()

        existing = make_exp_manager(log_dir="existing", is_existing_run=True)
        reused = make_tiny_trainer(exp=existing)
        assert not (
            pathlib.Path(cast(str, reused.log_dir)) / "run_config.yaml"
        ).exists()

    def test_checkpoint_path_triggers_load_checkpoint(
        self,
        tmp_path: Path,
        make_run_config: Callable[..., RunConfig],
        make_exp_manager: Callable[..., StubExpManager],
        make_batches: Callable[..., list[tuple[torch.Tensor, torch.Tensor]]],
        make_tiny_trainer: Callable[..., TinyTrainer],
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        mgr = CheckpointManager(str(ckpt_dir))
        src = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(src.parameters(), lr=0.1)
        mgr.save_checkpoint(3, src, opt, filename="resume.pth")
        mgr.close()

        rc = make_run_config(checkpoint_path=str(ckpt_dir / "resume.pth"))
        trainer = make_tiny_trainer(rc=rc, exp=make_exp_manager("resumed"))
        assert trainer._resumed is True
        assert trainer.start_epoch == 4  # saved epoch 3 -> next is 4
        assert torch.equal(trainer.model.weight.detach(), src.weight.detach())

    def test_second_build_warns_and_is_idempotent(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        warnings = []
        monkeypatch.setattr(
            base_trainer_module.logger,
            "warning",
            lambda msg, *a, **k: warnings.append(msg),
        )
        trainer = make_tiny_trainer()
        n_callbacks = len(_manager(trainer).callbacks)

        returned = trainer.build()

        assert returned is trainer
        assert trainer._built is True
        assert len(_manager(trainer).callbacks) == n_callbacks
        assert any("already built" in w for w in warnings)

    def test_compile_disabled_leaves_model_class(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()  # rc.compile.compile defaults to False
        assert type(trainer.model) is torch.nn.Linear


# ---------------------------------------------------------------------------
# run() lifecycle
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_run_without_build_raises(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()
        trainer._built = False
        with pytest.raises(RuntimeError, match="not been built"):
            trainer.run()

    def test_full_run_writes_metric_csvs_and_closes_resources(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        import pathlib

        trainer = make_tiny_trainer()
        trainer.run()

        log_dir = pathlib.Path(cast(str, trainer.log_dir))
        assert (log_dir / "metrics_train.csv").exists()
        assert (log_dir / "metrics_val.csv").exists()
        assert cast(CheckpointManager, trainer.checkpoint_manager)._closed is True
        local = _find_local_backend(trainer.metric_logger)
        assert local is not None and local._csv_files == {}  # finish() closed handles

    def test_forward_pass_failure_still_runs_finish(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trainer = make_tiny_trainer()
        rec_logger = _RecordingMetricLogger()
        trainer.metric_logger = cast(MetricLogger, rec_logger)

        original = trainer.forward_pass
        state = {"n": 0}

        def flaky(batch: tuple[Any, ...]) -> dict[str, Any]:
            state["n"] += 1
            if state["n"] == 3:  # first validation batch -> mid-run blow-up
                raise RuntimeError("boom mid-run")
            return original(batch)

        monkeypatch.setattr(trainer, "forward_pass", flaky)

        with pytest.raises(RuntimeError, match="boom mid-run"):
            trainer.run()

        assert rec_logger.calls[-1] == "finish"  # run()'s finally invariant
        assert cast(CheckpointManager, trainer.checkpoint_manager)._closed is True

    def test_stage_result_fields(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()  # 1 epoch, val data present
        result = trainer._run_training_loop()

        assert isinstance(result, StageResult)
        assert result.name is None  # single-stage training
        assert result.monitor == "auc"
        assert result.final_epoch == 0
        assert result.best_epoch == 0
        assert result.best_metric is not None


# ---------------------------------------------------------------------------
# epoch loop
# ---------------------------------------------------------------------------


class TestEpochLoop:
    def test_hook_order(self, make_tiny_trainer: Callable[..., TinyTrainer]) -> None:
        rec = _RecordingCallback()
        trainer = make_tiny_trainer(extra_callbacks=[rec])
        trainer.run()

        names = [e[0] for e in rec.events]
        assert names == [
            "train_begin",
            "epoch_begin",
            "phase_begin",
            "batch_begin",
            "batch_end",
            "batch_begin",
            "batch_end",
            "phase_end",
            "phase_begin",
            "batch_begin",
            "batch_end",
            "batch_begin",
            "batch_end",
            "phase_end",
            "epoch_end",
            "train_end",
        ]
        phases = [e[2] for e in rec.events if e[0] == "phase_begin"]
        assert phases == ["train", "val"]

    def test_scheduler_steps_once_per_epoch(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        sched = _CountingScheduler()
        trainer = make_tiny_trainer(lr_scheduler=sched, model_kwargs={"epochs": 3})
        trainer.run()
        assert sched.steps == 3

    def test_early_stopping_breaks_loop(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
    ) -> None:
        rc = make_run_config(
            early_stopping=EarlyStoppingConfig(patience=0),
            model_kwargs={"epochs": 3},
        )
        trainer = make_tiny_trainer(rc=rc)
        # Freeze the optimizer: identical val metrics every epoch, so the
        # second epoch is a bad epoch and patience=0 stops the loop.
        trainer.opt = torch.optim.SGD(trainer.model.parameters(), lr=0.0)

        rec = _RecordingCallback()
        _manager(trainer).callbacks.append(rec)
        trainer.run()

        epoch_ends = [e for e in rec.events if e[0] == "epoch_end"]
        assert len(epoch_ends) == 2  # epochs 0 and 1, not 3
        assert len(trainer._epoch_times) == 2

    def test_global_step_counts_train_batches_only(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()  # 2 train + 2 val batches, 1 epoch
        trainer.run()
        assert trainer._global_step == 2


# ---------------------------------------------------------------------------
# _process_epoch
# ---------------------------------------------------------------------------


class TestProcessEpoch:
    def test_sample_weighted_mean_loss(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        rec = _RecordingCallback()
        trainer = make_tiny_trainer(extra_callbacks=[rec])  # batch sizes 4 and 2
        trainer.run()

        batch_losses = [
            e[4] for e in rec.events if e[0] == "batch_end" and e[3] == "train"
        ]
        train_mean = next(
            e[3] for e in rec.events if e[0] == "phase_end" and e[2] == "train"
        )
        assert len(batch_losses) == 2
        expected = (batch_losses[0] * 4 + batch_losses[1] * 2) / 6
        assert train_mean == pytest.approx(expected)
        # ...and not the unweighted batch mean:
        assert train_mean != pytest.approx(sum(batch_losses) / 2)

    def test_model_mode_switches_per_phase(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen_modes = []
        trainer = make_tiny_trainer()
        original_bound = trainer.forward_pass

        def probe(batch: tuple[Any, ...]) -> dict[str, Any]:
            seen_modes.append(trainer.model.training)
            return original_bound(batch)

        monkeypatch.setattr(trainer, "forward_pass", probe)
        trainer.run()

        assert seen_modes[:2] == [True, True]  # train phase -> model.train()
        assert seen_modes[2:] == [False, False]  # val phase -> model.eval()

    def test_log_batch_metrics_writes_csv(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
    ) -> None:
        rc = make_run_config(log_batch_metrics=True, model_kwargs={"epochs": 1})
        trainer = make_tiny_trainer(rc=rc)
        trainer.run()

        path = pathlib.Path(cast(str, trainer.log_dir)) / "batch_metrics_train.csv"
        assert path.exists()
        rows = list(csv.reader(path.open(newline="")))
        assert rows[0] == ["global_step", "epoch", "batch_idx", "loss"]
        assert len(rows) == 3  # header + 2 train batches


# ---------------------------------------------------------------------------
# compute_train_step
# ---------------------------------------------------------------------------


class TestComputeTrainStep:
    def test_weights_change_after_step(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()
        weight_before = trainer.model.weight.detach().clone()

        output, loss = trainer.compute_train_step(trainer.train_data[0])

        assert set(output) >= {"y_hat", "y_label", "y_predict"}
        assert loss.item() >= 0.0
        assert not torch.equal(weight_before, trainer.model.weight.detach())

    def test_clip_skipped_when_max_clip_grad_norm_none(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []
        monkeypatch.setattr(
            torch.nn.utils, "clip_grad_norm_", lambda *a, **k: calls.append(a)
        )

        trainer = make_tiny_trainer()  # max_clip_grad_norm default None
        trainer.compute_train_step(trainer.train_data[0])
        assert calls == []

        clipped = make_tiny_trainer(max_clip_grad_norm=0.5)
        clipped.compute_train_step(clipped.train_data[0])
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _evaluate_on_test_set
# ---------------------------------------------------------------------------


class TestEvaluateOnTestSet:
    def test_best_weights_loaded_then_current_restored(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_batches: Callable[..., list[tuple[torch.Tensor, torch.Tensor]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trainer = make_tiny_trainer(test=make_batches(), model_kwargs={"epochs": 1})
        trainer.run()

        # Marker parameters: best state uses weight 9.0, current uses 1.0.
        checkpoint_cb = _required_cb(trainer, CheckpointCallback)
        best_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        with torch.no_grad():
            best_state["weight"].fill_(9.0)
            best_state["bias"].fill_(0.0)
        checkpoint_cb.best_model_state = best_state
        with torch.no_grad():
            trainer.model.weight.fill_(1.0)
            trainer.model.bias.fill_(0.0)

        seen_weights = []
        original_bound = trainer.forward_pass

        def probe(batch: tuple[Any, ...]) -> dict[str, Any]:
            seen_weights.append(trainer.model.weight.detach().clone())
            return original_bound(batch)

        monkeypatch.setattr(trainer, "forward_pass", probe)

        trainer._evaluate_on_test_set(use_best_model=True)

        assert seen_weights  # evaluation actually ran
        assert all(torch.all(w == 9.0) for w in seen_weights)  # best loaded
        assert torch.all(trainer.model.weight.detach() == 1.0)  # current restored

    def test_no_val_data_skips_val_phase(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        rec = _RecordingCallback()
        trainer = make_tiny_trainer(val=None, extra_callbacks=[rec])
        trainer.run()

        phases = [e[2] for e in rec.events if e[0] == "phase_begin"]
        assert phases == ["train"]
        assert not (
            pathlib.Path(cast(str, trainer.log_dir)) / "metrics_val.csv"
        ).exists()


# ---------------------------------------------------------------------------
# progress rendering gating
# ---------------------------------------------------------------------------


class TestProgressGating:
    def test_progress_none_omits_callback(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        from utils.training.callbacks import ProgressCallback

        trainer = make_tiny_trainer()  # conftest default: progress="none"
        assert _manager(trainer).get_callback(ProgressCallback) is None
        assert trainer._progress_enabled is False

    def test_progress_rich_adds_callback(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
    ) -> None:
        from utils.training.callbacks import ProgressCallback

        trainer = make_tiny_trainer(rc=make_run_config(progress="rich"))
        assert _manager(trainer).get_callback(ProgressCallback) is not None

    def test_progress_callback_precedes_test_evaluation(
        self,
        make_run_config: Callable[..., RunConfig],
        make_exp_manager: Callable[..., StubExpManager],
        make_batches: Callable[..., list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        from utils.training.callbacks import ProgressCallback

        trainer = TinyTrainer(
            make_run_config(progress="rich", skip_test=False),
            make_exp_manager(),
            make_batches(),
        )
        callbacks = _manager(trainer).callbacks
        progress_idx = next(
            i for i, cb in enumerate(callbacks) if isinstance(cb, ProgressCallback)
        )
        test_idx = next(
            i
            for i, cb in enumerate(callbacks)
            if isinstance(cb, TestEvaluationCallback)
        )
        # on_train_end order: the live display must stop before test
        # evaluation renders its own progress bar.
        assert progress_idx < test_idx

    def test_progress_callback_is_first_callback(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
    ) -> None:
        from utils.training.callbacks import ProgressCallback

        trainer = make_tiny_trainer(rc=make_run_config(progress="rich"))
        # First in the list: every later callback's on_train_end runs
        # after ProgressCallback.on_train_end stopped the live display.
        assert isinstance(_manager(trainer).callbacks[0], ProgressCallback)

    def test_custom_on_train_end_runs_after_display_teardown(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
    ) -> None:
        from utils.training.callbacks import ProgressCallback

        live_states = []

        class Probe(Callback):
            def on_train_end(self, **kwargs: Any) -> None:
                cb = kwargs["trainer"].callback_manager.get_callback(ProgressCallback)
                live_states.append(cb._live)

        trainer = make_tiny_trainer(
            rc=make_run_config(progress="rich"), extra_callbacks=[Probe()]
        )
        trainer.run()
        assert live_states == [None]  # display already stopped

    def test_rich_loop_end_to_end_tears_down(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
    ) -> None:
        from utils.training.callbacks import ProgressCallback

        trainer = make_tiny_trainer(rc=make_run_config(progress="rich"))
        result = trainer._run_training_loop()

        assert isinstance(result, StageResult)
        assert result.best_metric is not None
        cb = _required_cb(trainer, ProgressCallback)
        assert cb._live is None  # on_train_end stopped the display

    def test_failed_run_closes_progress_display(
        self,
        make_tiny_trainer: Callable[..., TinyTrainer],
        make_run_config: Callable[..., RunConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from utils.training.callbacks import ProgressCallback

        trainer = make_tiny_trainer(rc=make_run_config(progress="rich"))
        cb = _required_cb(trainer, ProgressCallback)

        original = trainer.forward_pass
        state = {"n": 0}

        def flaky(batch: tuple[Any, ...]) -> dict[str, Any]:
            state["n"] += 1
            if state["n"] == 3:  # first validation batch -> mid-run blow-up
                raise RuntimeError("boom mid-run")
            return original(batch)

        monkeypatch.setattr(trainer, "forward_pass", flaky)

        with pytest.raises(RuntimeError, match="boom mid-run"):
            trainer.run()

        # run()'s finally -> _finish -> CallbackManager.close() tore it down.
        assert cb._live is None

    def test_eval_batches_iterate_plain_when_disabled(
        self, make_tiny_trainer: Callable[..., TinyTrainer]
    ) -> None:
        trainer = make_tiny_trainer()
        batches = list(trainer._iter_eval_batches([1, 2, 3], "Testing"))
        assert batches == [1, 2, 3]
