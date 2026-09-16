"""Tests for ``ProgressCallback``: lifecycle, re-entry, and teardown.

Hooks are driven directly with a duck-typed trainer (area style: no
``unittest.mock``), mirroring ``test_callbacks.py``. Rich renders into
pytest's captured non-TTY stdout, which is fine — assertions target the
callback's own state, not console output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from rich.live import Live
from rich.progress import Progress, TaskID
from rich.text import Text

from utils.config.run_config import EarlyStoppingConfig
from utils.training.callbacks import (
    CallbackManager,
    CheckpointCallback,
    ProgressCallback,
)
from utils.training.checkpoint import CheckpointManager
from utils.training.early_stopping import EarlyStopping


class _StubTrainer:
    """Duck-typed trainer exposing exactly the attrs ProgressCallback reads."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        val_data: list[int] | None = None,
        early_stopping: EarlyStopping | None = None,
        stage: str | None = None,
    ) -> None:
        self.callback_manager = CallbackManager(
            [CheckpointCallback(checkpoint_manager=CheckpointManager(str(tmp_path)))]
        )
        self.epochs = 2
        self.start_epoch = 0
        self.train_data = [1, 2]  # any sized iterable
        self.val_data = val_data if val_data is not None else [1]
        self.early_stopping = early_stopping
        self._current_stage = stage

    def _monitor_name(self) -> str:
        return "auc"


def _stub(tmp_path: Path, *, with_es: bool = True, **kwargs: Any) -> _StubTrainer:
    es = EarlyStopping(EarlyStoppingConfig(patience=2)) if with_es else None
    return _StubTrainer(tmp_path, early_stopping=es, **kwargs)


class TestLifecycle:
    def test_train_begin_without_trainer_is_noop(self, tmp_path: Path) -> None:
        cb = ProgressCallback()
        cb.on_train_begin(2)
        assert cb._live is None

    def test_full_lifecycle_builds_updates_and_tears_down(self, tmp_path: Path) -> None:
        cb = ProgressCallback()
        trainer = _stub(tmp_path)
        cb.on_train_begin(2, trainer=trainer)

        assert cb._live is not None and cb._live.is_started
        assert cb._best_text is not None
        assert "Best AUC: N/A" in cb._best_text.plain

        cb.on_phase_begin(0, "train", trainer=trainer)
        cb.on_batch_end(0, 0, "train", 0.1, trainer=trainer)
        cb.on_batch_end(0, 1, "train", 0.2, trainer=trainer)

        checkpoint_cb = cast(
            CheckpointCallback,
            trainer.callback_manager.get_callback(CheckpointCallback),
        )
        checkpoint_cb.best_metric = 0.8421
        checkpoint_cb.best_epoch = 0
        cast(EarlyStopping, trainer.early_stopping).num_bad_epochs = 1
        cb.on_epoch_end(0, 0.5, 0.4, trainer=trainer)

        assert "Best AUC: 0.8421" in cb._best_text.plain
        assert "Patience: 1/2" in cb._best_text.plain

        cb.on_train_end(trainer=trainer)
        assert cb._live is None
        cb.close()  # idempotent
        assert cb._live is None

    def test_close_without_begin_is_noop(self) -> None:
        ProgressCallback().close()

    def test_no_early_stopping_skips_best_header(self, tmp_path: Path) -> None:
        cb = ProgressCallback()
        trainer = _stub(tmp_path, with_es=False)
        cb.on_train_begin(2, trainer=trainer)
        assert cb._best_text is None
        cb.on_epoch_end(0, 0.5, None, trainer=trainer)  # must not crash
        cb.on_train_end(trainer=trainer)

    def test_no_val_data_skips_best_header_refresh(self, tmp_path: Path) -> None:
        cb = ProgressCallback()
        trainer = _stub(tmp_path, val_data=None)
        cb.on_train_begin(2, trainer=trainer)
        # Header stays at its initial text; refresh path is gated on val data.
        cb.on_epoch_end(0, 0.5, None, trainer=trainer)
        assert "Best AUC: N/A" in cast(Text, cb._best_text).plain
        cb.on_train_end(trainer=trainer)


class TestMultiStageReentry:
    def test_second_train_begin_replaces_display(self, tmp_path: Path) -> None:
        cb = ProgressCallback()
        cb.on_train_begin(2, trainer=_stub(tmp_path, stage="km"))
        first_live = cast(Live, cb._live)

        cb.on_train_begin(2, trainer=_stub(tmp_path, stage="am"))

        assert first_live.is_started is False
        assert cb._live is not None and cb._live is not first_live
        assert cast(Text, cb._best_text).plain.startswith("[AM] Best AUC")
        cb.close()

    def test_phase_begin_reads_current_stage_loader(self, tmp_path: Path) -> None:
        cb = ProgressCallback()
        trainer = _stub(tmp_path)
        trainer.val_data = [1, 2, 3]
        cb.on_train_begin(2, trainer=trainer)

        cb.on_phase_begin(0, "val", trainer=trainer)
        progress = cast(Progress, cb._progress)
        work_task = cast("TaskID", cb._work_task)
        assert progress.tasks[work_task].total == 3
        cb.close()


class TestSingleDisplay:
    """rich allows one live display: duplicates must degrade, not crash."""

    def test_duplicate_registration_degrades_to_single_display(
        self, tmp_path: Path
    ) -> None:
        first, second = ProgressCallback(), ProgressCallback()
        first.on_train_begin(2, trainer=_stub(tmp_path))
        second.on_train_begin(2, trainer=_stub(tmp_path))

        assert first._live is None  # taken over, not left running
        assert second._live is not None and second._live.is_started
        second.close()

    def test_teardown_releases_the_active_slot(self, tmp_path: Path) -> None:
        cb = ProgressCallback()
        cb.on_train_begin(2, trainer=_stub(tmp_path))
        assert ProgressCallback._active is cb
        cb.close()
        assert ProgressCallback._active is None


@pytest.mark.parametrize(
    ("hook", "args"),
    [
        ("on_phase_begin", (0, "train")),
        ("on_batch_end", (0, 0, "train", 0.1)),
        ("on_epoch_end", (0, 0.5, 0.4)),
    ],
)
def test_hooks_before_begin_are_noop(hook: str, args: tuple[Any, ...]) -> None:
    """Every render hook must tolerate firing before a display exists."""
    cb = ProgressCallback()
    getattr(cb, hook)(*args)
    assert cb._live is None
