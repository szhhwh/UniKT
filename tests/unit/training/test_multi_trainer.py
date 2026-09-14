"""Tests for ``utils.training.multi_trainer``: stage lifecycle and wiring.

Runs the conftest ``TinyMultiTrainer`` (two tiny Linear stages over plain
``(x, y)`` tuple batches) end-to-end on CPU; stage ordering, lazy builds,
per-stage callbacks/early stopping, and metric-step bookkeeping are asserted
via the trainer's recorded event/snapshot logs.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.unit.training.conftest import TinyMultiTrainer
from utils.config.run_config import EarlyStoppingConfig
from utils.training.base_trainer import StageResult
from utils.training.callbacks import Callback


class _OffsetRecordingCallback(Callback):
    """Records trainer._metric_step_offset at every epoch begin."""

    def __init__(self):
        self.offsets = []

    def on_epoch_begin(self, epoch, **kwargs):
        self.offsets.append(kwargs["trainer"]._metric_step_offset)


class _EmptyStagesTrainer(TinyMultiTrainer):
    """build_stages() returning no stages must fail the run."""

    def build_stages(self):
        return []


class TestStageLifecycle:
    def test_empty_build_stages_raises_value_error(
        self, make_run_config, make_exp_manager, make_batches
    ):
        trainer = _EmptyStagesTrainer(
            make_run_config(), make_exp_manager(), make_batches()
        )
        with pytest.raises(ValueError, match="no stages"):
            trainer.run()

    def test_stages_built_lazily_in_order(
        self, make_run_config, make_exp_manager, make_batches
    ):
        trainer = TinyMultiTrainer(
            make_run_config(), make_exp_manager(), make_batches()
        )
        assert trainer.events == []  # construction builds no stage

        trainer.run()

        event_names = [(e[0], e[1]) for e in trainer.events]
        assert event_names == [
            ("begin", "km"),
            ("build", "km"),
            ("complete", "km"),
            ("begin", "am"),
            ("build", "am"),
            ("complete", "am"),
        ]
        # Lazy: the second stage is built only after the first completed.
        build_am_idx = event_names.index(("build", "am"))
        complete_km_idx = event_names.index(("complete", "km"))
        assert build_am_idx > complete_km_idx

    def test_callback_manager_rebuilt_per_stage(
        self, make_run_config, make_exp_manager, make_batches
    ):
        trainer = TinyMultiTrainer(
            make_run_config(), make_exp_manager(), make_batches()
        )
        trainer.run()
        managers = [
            snap["callback_manager"] for snap in trainer.stage_snapshots.values()
        ]
        assert managers[0] is not managers[1]

    def test_metric_step_offset_equals_previous_elapsed_epochs(
        self, make_run_config, make_exp_manager, make_batches
    ):
        rec = _OffsetRecordingCallback()
        specs = {
            "km": {"epochs": 2, "early_stopping": EarlyStoppingConfig(patience=2)},
            "am": {"epochs": 1, "early_stopping": EarlyStoppingConfig(patience=2)},
        }
        trainer = TinyMultiTrainer(
            make_run_config(),
            make_exp_manager(),
            make_batches(),
            stage_specs=specs,
            extra_callbacks=[rec],
        )
        trainer.run()

        assert rec.offsets == [0, 0, 2]  # km epochs 0-1, then am offset by 2

    def test_early_stopping_created_only_when_stage_config_present(
        self, make_run_config, make_exp_manager, make_batches
    ):
        specs = {
            "km": {"epochs": 1, "early_stopping": EarlyStoppingConfig(patience=2)},
            "am": {"epochs": 1},
        }
        trainer = TinyMultiTrainer(
            make_run_config(),
            make_exp_manager(),
            make_batches(),
            stage_specs=specs,
        )
        trainer.run()

        km, am = trainer.stage_snapshots["km"], trainer.stage_snapshots["am"]
        assert km["has_es"] and km["has_es_cb"]
        assert not am["has_es"] and not am["has_es_cb"]
        # Without early stopping there is no best-model file to track.
        assert km["best_filename"] == "best_km_model.pth"
        assert am["best_filename"] is None

    def test_checkpoint_monitor_mode_decoupled_from_early_stopping(
        self, make_run_config, make_exp_manager, make_batches
    ):
        specs = {
            "km": {
                "epochs": 1,
                "early_stopping": EarlyStoppingConfig(monitor="auc", mode="max"),
                "checkpoint_monitor": "rmse",
                "checkpoint_mode": "min",
            },
        }
        trainer = TinyMultiTrainer(
            make_run_config(),
            make_exp_manager(),
            make_batches(),
            stage_specs=specs,
        )
        trainer.run()

        snap = trainer.stage_snapshots["km"]
        assert snap["monitor_override"] == "rmse"
        assert snap["mode_override"] == "min"
        assert snap["es_monitor"] == "auc"  # early stopping keeps its own monitor
        assert trainer.early_stopping.cfg.mode == "max"

    def test_current_stage_cleared_after_loop(
        self, make_run_config, make_exp_manager, make_batches
    ):
        trainer = TinyMultiTrainer(
            make_run_config(), make_exp_manager(), make_batches()
        )
        trainer.run()
        assert trainer._current_stage is None
        assert set(trainer._stage_results) == {"km", "am"}

    def test_stage_hooks_receive_name_and_stage_result(
        self, make_run_config, make_exp_manager, make_batches
    ):
        trainer = TinyMultiTrainer(
            make_run_config(), make_exp_manager(), make_batches()
        )
        trainer.run()

        completes = [e for e in trainer.events if e[0] == "complete"]
        assert [e[1] for e in completes] == ["km", "am"]
        for name, result in ((e[1], e[2]) for e in completes):
            assert isinstance(result, StageResult)
            assert result.name == name
            assert result.final_epoch == 0

    def test_two_stage_end_to_end_run(
        self, make_run_config, make_exp_manager, make_batches
    ):
        exp = make_exp_manager("multi")
        trainer = TinyMultiTrainer(
            make_run_config(), exp, make_batches(), val=make_batches()
        )
        trainer.run()

        log_dir = pathlib.Path(exp.get_log_dir())
        for stage in ("km", "am"):
            assert (log_dir / f"metrics_{stage}_train.csv").exists()
            assert (log_dir / f"metrics_{stage}_val.csv").exists()
            assert (log_dir / f"best_{stage}_model.pth").exists()
        assert trainer.checkpoint_manager._closed is True

    def test_progress_callback_added_per_stage_when_rich(
        self, make_run_config, make_exp_manager, make_batches
    ):
        from utils.training.callbacks import ProgressCallback

        trainer = TinyMultiTrainer(
            make_run_config(progress="rich"),
            make_exp_manager("multi-rich"),
            make_batches(),
            val=make_batches(),
        )
        trainer.run()

        # Every stage's rebuilt manager owns its own display, each torn down.
        for snapshot in trainer.stage_snapshots.values():
            cb = snapshot["callback_manager"].get_callback(ProgressCallback)
            assert cb is not None
            assert cb._live is None
        assert trainer.stage_snapshots.keys() == {"km", "am"}

    def test_progress_callback_omitted_when_none(
        self, make_run_config, make_exp_manager, make_batches
    ):
        from utils.training.callbacks import ProgressCallback

        trainer = TinyMultiTrainer(
            make_run_config(),  # conftest default: progress="none"
            make_exp_manager("multi-none"),
            make_batches(),
            val=make_batches(),
        )
        trainer.run()

        for snapshot in trainer.stage_snapshots.values():
            assert snapshot["callback_manager"].get_callback(ProgressCallback) is None
