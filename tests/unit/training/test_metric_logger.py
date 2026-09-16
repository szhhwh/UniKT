"""Tests for ``utils.training.metric_logger``: local CSV, composite, async, factory.

Cloud backends (swanlab/wandb) are intercepted with sys.modules fakes from the
``inject_fake_cloud_logger`` fixture — both backends lazy-import their SDK
inside every method, so the fake is enough to observe init/log/finish traffic.
"""

from __future__ import annotations

import csv
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import pytest

from tests.unit.training.conftest import FakeCloudBackend
from utils.training.metric_logger import (
    AsyncMetricLoggerProxy,
    LocalMetricLogger,
    MetricLogger,
    MetricLoggerComposite,
    SwanLabMetricLogger,
    WandbMetricLogger,
    _async_enabled,
    _select_tracking_backend,
    build_default_metric_loggers,
    get_metric_logger,
)


def _rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


class _RecordingLogger:
    """Duck-typed backend recording (method, kwargs) for every call."""

    def __init__(
        self, fail: Iterable[str] = (), fail_exc: type[Exception] = RuntimeError
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._fail = set(fail)
        self._fail_exc = fail_exc

    def _rec(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, kwargs))
        if method in self._fail:
            raise self._fail_exc(f"{method} failed")

    def init_run(self, **kwargs: Any) -> None:
        self._rec("init_run", kwargs)

    def log_metrics(self, **kwargs: Any) -> None:
        self._rec("log_metrics", kwargs)

    def log_early_stopping(self, **kwargs: Any) -> None:
        self._rec("log_early_stopping", kwargs)

    def log_batch(self, **kwargs: Any) -> None:
        self._rec("log_batch", kwargs)

    def log_final(self, **kwargs: Any) -> None:
        self._rec("log_final", kwargs)

    def log_timing(self, **kwargs: Any) -> None:
        self._rec("log_timing", kwargs)

    def finish(self) -> None:
        self._rec("finish", {})


class _ThreadRecordingLogger:
    """Records the thread each method ran on."""

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def init_run(self, **kwargs: Any) -> None:
        self.events.append(("init_run", threading.get_ident()))

    def log_metrics(self, **kwargs: Any) -> None:
        self.events.append(("log_metrics", threading.get_ident()))

    def log_final(self, **kwargs: Any) -> None:
        self.events.append(("log_final", threading.get_ident()))

    def finish(self) -> None:
        self.events.append(("finish", threading.get_ident()))


def _unwrap(logger: object) -> object:
    return logger._inner if isinstance(logger, AsyncMetricLoggerProxy) else logger


def _backends(logger: object) -> list[Any]:
    return logger._loggers if isinstance(logger, MetricLoggerComposite) else [logger]


def _as_metric_logger(logger: object) -> MetricLogger:
    """The recording doubles duck-type the MetricLogger interface."""
    return cast(MetricLogger, logger)


# ---------------------------------------------------------------------------
# LocalMetricLogger
# ---------------------------------------------------------------------------


class TestLocalMetricLogger:
    def test_first_write_creates_sorted_header(self, tmp_path: Path) -> None:
        logger = LocalMetricLogger(log_dir=str(tmp_path))
        logger.log_metrics(
            phase="train",
            metrics={"rmse": 0.3, "acc": 0.9, "auc": 0.8},
            step=0,
            epoch=0,
        )
        logger.finish()

        rows = _rows(tmp_path / "metrics_train.csv")
        assert rows[0] == ["epoch", "acc", "auc", "rmse"]  # metric columns sorted
        assert rows[1] == ["0", "0.9", "0.8", "0.3"]

    def test_missing_keys_written_blank(self, tmp_path: Path) -> None:
        logger = LocalMetricLogger(log_dir=str(tmp_path))
        logger.log_metrics(
            phase="train", metrics={"acc": 0.9, "auc": 0.8}, step=0, epoch=0
        )
        logger.log_metrics(phase="train", metrics={"acc": 0.5}, step=1, epoch=1)
        logger.finish()

        rows = _rows(tmp_path / "metrics_train.csv")
        assert rows[2] == ["1", "0.5", ""]  # auc absent -> empty cell

    def test_new_columns_extend_header_in_place(self, tmp_path: Path) -> None:
        # A metric column appearing after the header was written must extend
        # the file: existing rows are back-filled with blanks under it.
        logger = LocalMetricLogger(log_dir=str(tmp_path))
        logger.log_metrics(phase="train", metrics={"acc": 0.9}, step=0, epoch=0)
        logger.log_metrics(
            phase="train", metrics={"acc": 0.5, "new_col": 1.0}, step=1, epoch=1
        )
        logger.finish()

        rows = _rows(tmp_path / "metrics_train.csv")
        assert rows[0] == ["epoch", "acc", "new_col"]
        assert rows[1] == ["0", "0.9", ""]  # earlier row back-filled blank
        assert rows[2] == ["1", "0.5", "1.0"]

    def test_none_metric_values_filtered_from_row(self, tmp_path: Path) -> None:
        logger = LocalMetricLogger(log_dir=str(tmp_path))
        logger.log_metrics(
            phase="val",
            metrics=cast("dict[str, float]", {"auc": None, "acc": 1.0}),
            step=0,
            epoch=0,
        )
        logger.finish()

        rows = _rows(tmp_path / "metrics_val.csv")
        assert rows[0] == ["epoch", "acc"]  # None metric never gains a column

    def test_stage_prefixed_filenames(self, tmp_path: Path) -> None:
        logger = LocalMetricLogger(log_dir=str(tmp_path))
        logger.log_metrics(
            phase="val", metrics={"auc": 0.7}, step=0, epoch=0, stage="km"
        )
        logger.log_metrics(phase="val", metrics={"auc": 0.7}, step=0, epoch=0)
        logger.log_early_stopping(
            phase="val",
            best_score=0.7,
            num_bad_epochs=0,
            best_metrics=None,
            step=0,
            epoch=0,
            stage="km",
        )
        logger.log_early_stopping(
            phase="val",
            best_score=0.7,
            num_bad_epochs=0,
            best_metrics=None,
            step=0,
            epoch=0,
        )
        logger.log_timing(step=0, epoch=0, timings={"train_time": 1.0}, stage="km")
        logger.log_timing(step=0, epoch=0, timings={"train_time": 1.0})
        logger.finish()

        assert (tmp_path / "metrics_km_val.csv").exists()
        assert (tmp_path / "metrics_val.csv").exists()
        assert (tmp_path / "early_stopping_km.csv").exists()
        assert (tmp_path / "early_stopping.csv").exists()
        assert (tmp_path / "timing_km.csv").exists()
        assert (tmp_path / "timing.csv").exists()

    def test_log_batch_gated_on_ctor_flag(self, tmp_path: Path) -> None:
        logger = LocalMetricLogger(log_dir=str(tmp_path), log_batch_metrics=False)
        logger.log_batch(phase="train", global_step=0, epoch=0, batch_idx=0, loss=0.5)
        logger.finish()
        assert not (tmp_path / "batch_metrics_train.csv").exists()

        enabled = LocalMetricLogger(log_dir=str(tmp_path), log_batch_metrics=True)
        enabled.log_batch(phase="train", global_step=3, epoch=1, batch_idx=2, loss=0.25)
        enabled.finish()
        rows = _rows(tmp_path / "batch_metrics_train.csv")
        assert rows[0] == ["global_step", "epoch", "batch_idx", "loss"]
        assert rows[1] == ["3", "1", "2", "0.25"]

    def test_log_final_leads_with_step_and_skips_empty(self, tmp_path: Path) -> None:
        logger = LocalMetricLogger(log_dir=str(tmp_path))
        logger.log_final(metrics={}, step=5)
        logger.log_final(metrics={"km_best": 0.8}, step=5)
        logger.finish()

        assert _rows(tmp_path / "metrics_final.csv")[0] == ["step", "km_best"]
        assert (
            len(_rows(tmp_path / "metrics_final.csv")) == 2
        )  # empty call wrote nothing

    def test_finish_idempotent(self, tmp_path: Path) -> None:
        logger = LocalMetricLogger(log_dir=str(tmp_path))
        logger.log_metrics(phase="train", metrics={"acc": 1.0}, step=0, epoch=0)
        logger.finish()
        logger.finish()  # second close of already-cleared handles is a no-op
        assert logger._csv_files == {}


# ---------------------------------------------------------------------------
# MetricLoggerComposite
# ---------------------------------------------------------------------------


class TestComposite:
    def test_fan_out_to_all_backends(self) -> None:
        first, second = _RecordingLogger(), _RecordingLogger()
        composite = MetricLoggerComposite(
            [_as_metric_logger(first), _as_metric_logger(second)]
        )

        composite.log_metrics(phase="train", metrics={"acc": 1.0}, step=0, epoch=0)
        composite.log_final(metrics={"x": 1.0}, step=1)
        composite.finish()

        for backend in (first, second):
            assert [c[0] for c in backend.calls] == [
                "log_metrics",
                "log_final",
                "finish",
            ]
            assert backend.calls[0][1] == {
                "phase": "train",
                "metrics": {"acc": 1.0},
                "step": 0,
                "epoch": 0,
            }

    def test_failing_backend_does_not_abort_others(self) -> None:
        failing, healthy = _RecordingLogger(fail={"log_metrics"}), _RecordingLogger()
        composite = MetricLoggerComposite(
            [_as_metric_logger(failing), _as_metric_logger(healthy)]
        )

        composite.log_metrics(phase="train", metrics={}, step=0, epoch=0)

        assert (
            "log_metrics",
            {"phase": "train", "metrics": {}, "step": 0, "epoch": 0},
        ) in healthy.calls

    def test_init_run_failure_isolated(self) -> None:
        failing, healthy = _RecordingLogger(fail={"init_run"}), _RecordingLogger()
        composite = MetricLoggerComposite(
            [_as_metric_logger(failing), _as_metric_logger(healthy)]
        )

        composite.init_run(
            log_dir="x", experiment_name="e", group="g", tags=[], config={}
        )

        assert any(method == "init_run" for method, _ in healthy.calls)

    def test_import_error_propagates_instead_of_isolating(self) -> None:
        # A missing dependency fails every future call from that backend;
        # isolating it would silently drop the backend's records behind a
        # single warning.
        failing = _RecordingLogger(fail={"log_metrics"}, fail_exc=ImportError)
        composite = MetricLoggerComposite(
            [_as_metric_logger(failing), _as_metric_logger(_RecordingLogger())]
        )

        with pytest.raises(ImportError):
            composite.log_metrics(phase="train", metrics={}, step=0, epoch=0)


# ---------------------------------------------------------------------------
# AsyncMetricLoggerProxy
# ---------------------------------------------------------------------------


class TestAsyncProxy:
    def test_init_run_runs_synchronously_on_main_thread(self) -> None:
        inner = _ThreadRecordingLogger()
        proxy = AsyncMetricLoggerProxy(_as_metric_logger(inner))

        proxy.init_run(experiment_name="e")

        assert inner.events[0][0] == "init_run"
        assert inner.events[0][1] == threading.get_ident()  # main thread

    def test_finish_flushes_pending_then_finishes_inner(self) -> None:
        inner = _ThreadRecordingLogger()
        proxy = AsyncMetricLoggerProxy(_as_metric_logger(inner))

        proxy.log_metrics(phase="train", metrics={"acc": 1.0}, step=0, epoch=0)
        proxy.finish()

        assert [e[0] for e in inner.events] == ["log_metrics", "finish"]

    def test_log_runs_on_worker_thread(self) -> None:
        inner = _ThreadRecordingLogger()
        proxy = AsyncMetricLoggerProxy(_as_metric_logger(inner))

        proxy.log_metrics(phase="train", metrics={}, step=0, epoch=0)
        proxy.flush()

        assert inner.events[0][1] != threading.get_ident()

    def test_submit_after_close_falls_back_to_sync(self) -> None:
        inner = _ThreadRecordingLogger()
        proxy = AsyncMetricLoggerProxy(_as_metric_logger(inner))
        proxy.close()

        proxy.log_metrics(phase="train", metrics={}, step=0, epoch=0)

        assert [e[0] for e in inner.events] == ["log_metrics"]  # ran inline
        assert proxy._futures == []

    def test_close_idempotent_and_does_not_finish_inner(self) -> None:
        inner = _ThreadRecordingLogger()
        proxy = AsyncMetricLoggerProxy(_as_metric_logger(inner))

        proxy.close()
        proxy.close()

        assert inner.events == []  # close drains only; finish stays with finish()

    def test_check_completed_prunes_finished_futures(self) -> None:
        inner = _ThreadRecordingLogger()
        proxy = AsyncMetricLoggerProxy(_as_metric_logger(inner))
        for i in range(3):
            proxy.log_final(metrics={"i": i}, step=i)

        time.sleep(0.2)  # let the worker drain the queue
        proxy.log_final(metrics={"i": 3}, step=3)

        assert len(proxy._futures) == 1  # only the last submit still tracked
        proxy.close()


# ---------------------------------------------------------------------------
# factory: get_metric_logger / env selection / build_default_metric_loggers
# ---------------------------------------------------------------------------


class TestFactory:
    def test_get_metric_logger_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            get_metric_logger("no-such-backend")

    def test_get_metric_logger_local(self, tmp_path: Path) -> None:
        logger = get_metric_logger("local", log_dir=str(tmp_path))
        assert isinstance(logger, LocalMetricLogger)

    def test_select_tracking_backend_default_and_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KT_TRACKING_BACKEND", raising=False)
        assert _select_tracking_backend() == "swanlab"
        monkeypatch.setenv("KT_TRACKING_BACKEND", "wandb")
        assert _select_tracking_backend() == "wandb"
        monkeypatch.setenv("KT_TRACKING_BACKEND", "  WANDB ")  # case + strip
        assert _select_tracking_backend() == "wandb"
        monkeypatch.setenv("KT_TRACKING_BACKEND", "")
        assert _select_tracking_backend() == "swanlab"  # empty falls back

    def test_select_tracking_backend_unknown_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KT_TRACKING_BACKEND", "tensorboard")
        with pytest.raises(ValueError, match="Unknown KT_TRACKING_BACKEND"):
            _select_tracking_backend()

    def test_async_enabled_default_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("METRIC_LOGGING_ASYNC", raising=False)
        assert _async_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_async_enabled_disabled_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("METRIC_LOGGING_ASYNC", value)
        assert _async_enabled() is False

    def test_build_default_local_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("METRIC_LOGGING_ASYNC", raising=False)
        logger = build_default_metric_loggers(
            log_dir=str(tmp_path), log_batch_metrics=False, cloud_tracking=False
        )

        backends = [_unwrap(b) for b in _backends(logger)]
        assert [type(b) for b in backends] == [LocalMetricLogger]

    def test_build_default_missing_cloud_backend_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KT_TRACKING_BACKEND", raising=False)
        # A None sys.modules entry makes importlib.import_module raise
        # ImportError without uninstalling the real package.
        monkeypatch.setitem(sys.modules, "swanlab", None)

        with pytest.raises(RuntimeError, match="not importable"):
            build_default_metric_loggers(
                log_dir=str(tmp_path), log_batch_metrics=False, cloud_tracking=True
            )

    def test_build_default_async_disabled_leaves_raw_backends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        explicit = build_default_metric_loggers(
            log_dir=str(tmp_path),
            log_batch_metrics=False,
            cloud_tracking=False,
            async_io=False,
        )
        assert all(
            not isinstance(b, AsyncMetricLoggerProxy) for b in _backends(explicit)
        )

        monkeypatch.setenv("METRIC_LOGGING_ASYNC", "0")
        env_driven = build_default_metric_loggers(
            log_dir=str(tmp_path), log_batch_metrics=False, cloud_tracking=False
        )
        assert all(
            not isinstance(b, AsyncMetricLoggerProxy) for b in _backends(env_driven)
        )

    def test_build_default_with_fake_swanlab(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        inject_fake_cloud_logger: Callable[[str], FakeCloudBackend],
    ) -> None:
        monkeypatch.delenv("KT_TRACKING_BACKEND", raising=False)
        monkeypatch.delenv("LARK_WEBHOOK_URL", raising=False)
        backend = inject_fake_cloud_logger("swanlab")

        logger = build_default_metric_loggers(
            log_dir=str(tmp_path),
            log_batch_metrics=False,
            cloud_tracking=True,
            async_io=False,
        )
        kinds = {type(b) for b in _backends(logger)}
        assert kinds == {LocalMetricLogger, SwanLabMetricLogger}

        logger.init_run(
            log_dir=str(tmp_path),
            experiment_name="exp1",
            group="g",
            tags=["cpu"],
            config={},
        )
        init_calls = [c for c in backend.calls if c[0] == "init"]
        assert len(init_calls) == 1
        assert init_calls[0][1]["name"] == "Run_exp1"

        logger.log_metrics(phase="train", metrics={"acc": 1.0}, step=0, epoch=0)
        logger.finish()
        assert any(c[0] == "log" for c in backend.calls)
        assert any(c[0] == "finish" for c in backend.calls)

    def test_build_default_with_fake_wandb(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        inject_fake_cloud_logger: Callable[[str], FakeCloudBackend],
    ) -> None:
        monkeypatch.setenv("KT_TRACKING_BACKEND", "wandb")
        monkeypatch.delenv("KT_WANDB_PROJECT", raising=False)
        backend = inject_fake_cloud_logger("wandb")

        logger = build_default_metric_loggers(
            log_dir=str(tmp_path),
            log_batch_metrics=False,
            cloud_tracking=True,
            async_io=False,
        )
        kinds = {type(b) for b in _backends(logger)}
        assert kinds == {LocalMetricLogger, WandbMetricLogger}

        logger.init_run(
            log_dir=str(tmp_path),
            experiment_name="exp2",
            group="g",
            tags=["cpu"],
            config={},
        )
        init_calls = [c for c in backend.calls if c[0] == "init"]
        assert init_calls[0][1]["project"] == "UniKT"
        assert init_calls[0][1]["name"] == "exp2"

        logger.finish()
        assert any(c[0] == "finish" for c in backend.calls)
