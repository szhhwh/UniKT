"""Metric logging backend module.

Provides a unified metric logging abstraction (MetricLogger) consistent
with the project's registry + ABC + factory pattern (see DataSource /
DATA_SOURCES / get_data_source).

Backends:
- LocalMetricLogger: Local CSV logging, always enabled.
- SwanLabMetricLogger: SwanLab remote logging.
- WandbMetricLogger: Weights & Biases remote logging.

Cloud tracking (SwanLab or W&B) is enabled by ``--general.cloud_tracking``
(default on); the two are mutually exclusive — ``KT_TRACKING_BACKEND``
selects which (``swanlab`` | ``wandb``, default ``swanlab``).
"""

import atexit
import csv
import importlib
import os
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import IO, Any

from ..core import METRIC_LOGGERS, get_logger, register_metric_logger

logger = get_logger(__name__)


class MetricLogger(ABC):
    """Abstract base class for metric logging backends.

    All methods must be safe to call repeatedly. Backend initialization
    (e.g. swanlab.init) is deferred to ``init_run``; ``log_*`` methods
    should silently skip when not initialized.
    """

    @abstractmethod
    def init_run(
        self,
        *,
        log_dir: str,
        experiment_name: str,
        group: str,
        tags: list[str],
        config: dict[str, Any],
    ) -> None:
        """Initialize the logging backend for a new run.

        Args:
            log_dir: Directory for storing logs.
            experiment_name: Name of this experiment.
            group: Experiment group for grouping runs.
            tags: Tags for categorizing the run.
            config: Experiment configuration dict.
        """

    @abstractmethod
    def log_metrics(
        self,
        *,
        phase: str,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log epoch-level aggregated metrics.

        Args:
            phase: Phase name (e.g. ``"train"``, ``"val"``).
            metrics: Metric name to value mapping.
            step: Global step number.
            epoch: Current epoch number.
            stage: Stage name for multi-stage training (optional).
        """

    @abstractmethod
    def log_early_stopping(
        self,
        *,
        phase: str,
        best_score: float | None,
        num_bad_epochs: int,
        best_metrics: dict[str, float] | None,
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log early stopping trajectory.

        Args:
            phase: Phase name.
            best_score: Best monitored score so far.
            num_bad_epochs: Number of epochs without improvement.
            best_metrics: Best metric values (optional).
            step: Global step number.
            epoch: Current epoch number.
            stage: Stage name (optional).
        """

    @abstractmethod
    def log_batch(
        self,
        *,
        phase: str,
        global_step: int,
        epoch: int,
        batch_idx: int,
        loss: float,
        stage: str | None = None,
    ) -> None:
        """Log per-batch loss.

        Args:
            phase: Phase name.
            global_step: Global step number.
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            loss: Loss value for this batch.
            stage: Stage name (optional).
        """

    @abstractmethod
    def log_final(self, *, metrics: dict[str, float], step: int) -> None:
        """Log final summary metrics.

        Args:
            metrics: Final metric values.
            step: Global step number.
        """

    @abstractmethod
    def log_timing(
        self,
        *,
        step: int,
        epoch: int,
        timings: dict[str, float],
        stage: str | None = None,
    ) -> None:
        """Log per-epoch timing breakdown (train/val/total) for a stage."""

    @abstractmethod
    def finish(self) -> None:
        """Clean up and tear down the logging backend."""


@register_metric_logger("local")
class LocalMetricLogger(MetricLogger):
    """Local CSV metric logger.

    Writes CSV files under ``log_dir`` per phase:
      ``metrics_{phase}.csv``           Epoch-level aggregated metrics.
      ``early_stopping[_{stage}].csv``   Early stopping trajectory.
      ``batch_metrics_{phase}.csv``      Per-batch loss (when enabled).
      ``metrics_final.csv``              Final summary metrics.

    Multi-stage runs distinguish series via ``metrics_{stage}_{phase}.csv``.
    Headers are written lazily and extended in-place when new metric
    columns appear.
    """

    def __init__(self, *, log_dir: str, log_batch_metrics: bool = False):
        """Initialize the local CSV metric logger.

        Args:
            log_dir: Directory for output CSV files.
            log_batch_metrics: Whether to log per-batch loss.
        """
        self._log_dir = log_dir
        self._log_batch = log_batch_metrics
        os.makedirs(log_dir, exist_ok=True)
        self._csv_files: dict[str, IO] = {}
        self._csv_headers: dict[str, list[str]] = {}

    @staticmethod
    def _series(phase: str, stage: str | None) -> str:
        """Build a series identifier from phase and optional stage."""
        return f"{stage}_{phase}" if stage else phase

    def init_run(self, **kwargs: Any) -> None:
        """No-op: log_dir was already set at construction time."""
        pass

    def _write_row(
        self, path: str, leading: list[tuple[str, Any]], values: dict[str, Any]
    ) -> None:
        """Write a single CSV row.

        ``leading`` columns are fixed prefix pairs ``(name, value)``,
        ``values`` are dynamic metric columns. The header is determined
        on first write; subsequent rows align to existing columns, with
        missing values left blank. A metric column appearing for the first
        time after the header was written extends the file in-place: all
        existing rows are back-filled with blanks under the new column.

        Args:
            path: CSV file path.
            leading: Ordered list of ``(column_name, value)`` prefix pairs.
            values: Dynamic metric column name to value mapping.
        """
        f = self._csv_files.get(path)
        if f is None:
            header = [c for c, _ in leading] + sorted(values.keys())
            f = open(path, "a", newline="")  # noqa: SIM115
            self._csv_files[path] = f
            self._csv_headers[path] = header
            csv.writer(f).writerow(header)
        metric_cols = self._csv_headers[path][len(leading) :]
        new_cols = sorted(set(values) - set(metric_cols))
        if new_cols:
            metric_cols = self._extend_header(path, leading, metric_cols, new_cols)
            f = self._csv_files[path]
        row = [v for _, v in leading] + [values.get(c, "") for c in metric_cols]
        csv.writer(f).writerow(row)
        f.flush()

    def _extend_header(
        self,
        path: str,
        leading: list[tuple[str, Any]],
        metric_cols: list[str],
        new_cols: list[str],
    ) -> list[str]:
        """Rewrite the file with additional metric columns (blanks back-filled)."""
        self._csv_files[path].close()
        with open(path, newline="") as f:
            existing = list(csv.reader(f))
        extended = metric_cols + new_cols
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([c for c, _ in leading] + extended)
            for row in existing[1:]:
                writer.writerow(row + [""] * len(new_cols))
        self._csv_files[path] = open(path, "a", newline="")  # noqa: SIM115
        self._csv_headers[path] = [c for c, _ in leading] + extended
        return extended

    def log_metrics(
        self,
        *,
        phase: str,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log epoch-level metrics to a CSV file.

        Args:
            phase: Phase name (e.g. ``"train"``, ``"val"``).
            metrics: Metric name to value mapping.
            step: Global step (unused in CSV logging).
            epoch: Current epoch number.
            stage: Stage name (optional).
        """
        values = {k: v for k, v in metrics.items() if v is not None}
        path = os.path.join(self._log_dir, f"metrics_{self._series(phase, stage)}.csv")
        self._write_row(path, [("epoch", epoch)], values)

    def log_early_stopping(
        self,
        *,
        phase: str,
        best_score: float | None,
        num_bad_epochs: int,
        best_metrics: dict[str, float] | None,
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log early stopping trajectory to a CSV file.

        Args:
            phase: Phase name.
            best_score: Best monitored score.
            num_bad_epochs: Consecutive epochs without improvement.
            best_metrics: Best metric values (optional).
            step: Global step (unused in CSV logging).
            epoch: Current epoch number.
            stage: Stage name (optional).
        """
        values: dict[str, Any] = {
            "best_score": best_score if best_score is not None else "",
            "num_bad_epochs": num_bad_epochs,
        }
        if best_metrics:
            values.update({f"best_{k}": v for k, v in best_metrics.items()})
        filename = f"early_stopping_{stage}.csv" if stage else "early_stopping.csv"
        path = os.path.join(self._log_dir, filename)
        self._write_row(path, [("epoch", epoch)], values)

    def log_batch(
        self,
        *,
        phase: str,
        global_step: int,
        epoch: int,
        batch_idx: int,
        loss: float,
        stage: str | None = None,
    ) -> None:
        """Log per-batch loss to a CSV file.

        Only writes when ``log_batch_metrics`` was enabled at init.

        Args:
            phase: Phase name.
            global_step: Global step number.
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            loss: Loss value for this batch.
            stage: Stage name (optional).
        """
        if not self._log_batch:
            return
        path = os.path.join(
            self._log_dir, f"batch_metrics_{self._series(phase, stage)}.csv"
        )
        self._write_row(
            path,
            [
                ("global_step", global_step),
                ("epoch", epoch),
                ("batch_idx", batch_idx),
                ("loss", loss),
            ],
            {},
        )

    def log_final(self, *, metrics: dict[str, float], step: int) -> None:
        """Log final summary metrics to a CSV file.

        Args:
            metrics: Final metric values.
            step: Global step number.
        """
        if not metrics:
            return
        path = os.path.join(self._log_dir, "metrics_final.csv")
        self._write_row(path, [("step", step)], dict(metrics))

    def log_timing(
        self,
        *,
        step: int,
        epoch: int,
        timings: dict[str, float],
        stage: str | None = None,
    ) -> None:
        """Log per-epoch timing breakdown (train/val/total) for a stage."""
        filename = f"timing_{stage}.csv" if stage else "timing.csv"
        path = os.path.join(self._log_dir, filename)
        self._write_row(path, [("epoch", epoch)], dict(timings))

    def finish(self) -> None:
        """Close all open CSV file handles."""
        for f in self._csv_files.values():
            with suppress(Exception):
                f.close()
        self._csv_files.clear()
        self._csv_headers.clear()


@register_metric_logger("swanlab")
class SwanLabMetricLogger(MetricLogger):
    """SwanLab metric logging backend.

    ``swanlab`` is lazily imported inside each method to keep it an
    optional dependency.
    """

    def __init__(self) -> None:
        """Initialize the SwanLab logger with uninitialized state."""
        self._initialized = False

    def init_run(
        self,
        *,
        log_dir: str,
        experiment_name: str,
        group: str,
        tags: list[str],
        config: dict[str, Any],
    ) -> None:
        """Initialize the SwanLab run.

        Args:
            log_dir: Log directory (passed through to swanlab).
            experiment_name: Name of this experiment.
            group: Experiment group.
            tags: Tags for the run.
            config: Experiment configuration dict.
        """
        import swanlab
        from swanlab.exceptions import AuthenticationError
        from swanlab.plugin.notification import LarkCallback

        callbacks = []
        webhook = os.getenv("LARK_WEBHOOK_URL")
        secret = os.getenv("LARK_SECRET")
        if webhook:
            callbacks.append(LarkCallback(webhook_url=webhook, secret=secret))

        # Re-authenticate per run: SwanLab revokes the session token on
        # finish() but the SDK reuses the now-invalid client, so the next run
        # in the same process would 401 on every request. Only re-authenticate
        # when a stored key exists — without one, login() raises
        # AuthenticationError and would short-circuit swanlab.init's own
        # interactive login prompt (prompt_init_mode), the only path that
        # actually surfaces an API-key input box.
        try:
            swanlab.login(relogin=True)
        except AuthenticationError:
            logger.info(
                "No stored SwanLab API key; deferring to interactive login "
                "inside swanlab.init."
            )
        # reinit finalizes any still-active run (e.g. a prior trial that raised
        # before finishing) before starting this one, instead of erroring out.
        swanlab.init(
            workspace=os.getenv("KT_SWANLAB_WORKSPACE") or None,
            project=os.getenv("KT_SWANLAB_PROJECT") or "UniKT",
            name=f"Run_{experiment_name}",
            config=config,
            callbacks=callbacks,
            group=group,
            tags=tags,
            settings=swanlab.Settings(),
            reinit=True,
        )
        self._initialized = True

    @staticmethod
    def _prefix(phase: str, stage: str | None) -> str:
        """Build a SwanLab metric prefix from phase and optional stage."""
        return (
            f"{stage.upper()}/{phase.capitalize()}/"
            if stage
            else f"{phase.capitalize()}/"
        )

    def log_metrics(
        self,
        *,
        phase: str,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log epoch-level metrics to SwanLab.

        Args:
            phase: Phase name.
            metrics: Metric name to value mapping.
            step: Global step number.
            epoch: Current epoch number (unused in SwanLab).
            stage: Stage name (optional).
        """
        if not self._initialized:
            return
        import swanlab

        prefix = self._prefix(phase, stage)
        payload = {
            f"{prefix}{name.upper()}-epoch": v
            for name, v in metrics.items()
            if v is not None
        }
        if payload:
            swanlab.log(payload, step=step)

    def log_early_stopping(
        self,
        *,
        phase: str,
        best_score: float | None,
        num_bad_epochs: int,
        best_metrics: dict[str, float] | None,
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log early stopping trajectory to SwanLab.

        Args:
            phase: Phase name.
            best_score: Best monitored score.
            num_bad_epochs: Consecutive epochs without improvement.
            best_metrics: Best metric values (optional).
            step: Global step number.
            epoch: Current epoch number (unused in SwanLab).
            stage: Stage name (optional).
        """
        if not self._initialized:
            return
        import swanlab

        sp = f"{stage.upper()}/" if stage else ""
        data = {
            f"{sp}ES/Best": best_score,
            f"{sp}ES/Num_Bad_Epochs": num_bad_epochs,
        }
        if best_metrics:
            data.update(
                {f"{sp}ES/Best_{k.upper()}": v for k, v in best_metrics.items()}
            )
        swanlab.log(data, step=step)

    def log_batch(self, **kwargs: Any) -> None:
        """No-op: SwanLab does not log per-batch metrics to avoid noise."""

    def log_final(self, *, metrics: dict[str, float], step: int) -> None:
        """Log final summary metrics to SwanLab.

        Args:
            metrics: Final metric values.
            step: Global step number.
        """
        if not self._initialized or not metrics:
            return
        import swanlab

        swanlab.log(metrics, step=step)

    def log_timing(
        self,
        *,
        step: int,
        epoch: int,
        timings: dict[str, float],
        stage: str | None = None,
    ) -> None:
        """Log per-epoch timing breakdown to SwanLab for a stage."""
        if not self._initialized:
            return
        import swanlab

        sp = f"{stage.upper()}/" if stage else ""
        data = {
            f"{sp}Time/{k.replace('_time', '').title()}": v
            for k, v in timings.items()
            if v is not None
        }
        if data:
            swanlab.log(data, step=step)

    def finish(self) -> None:
        """Finish the SwanLab run and clean up."""
        if not self._initialized:
            return
        import swanlab

        swanlab.finish()
        self._initialized = False


@register_metric_logger("wandb")
class WandbMetricLogger(MetricLogger):
    """Weights & Biases metric logging backend.

    ``wandb`` is lazily imported inside each method to keep it an
    optional dependency at import time.

    Step handling: W&B rejects non-monotonic ``step`` values (it drops the
    offending log call), but our callers pass two incompatible scales —
    ``log_metrics``/``log_timing``/test use an epoch-based step while
    ``log_early_stopping`` (callbacks.py) uses the batch-level
    ``_global_step``. Interleaving them makes the step jump backward every
    epoch, so W&B would silently drop almost every record. We therefore
    ignore the external ``step`` and advance our own counter on every
    commit. (SwanLab tolerates non-monotonic steps, so SwanLabMetricLogger
    forwards the raw value unchanged.)
    """

    def __init__(self) -> None:
        """Initialize the W&B logger with uninitialized state."""
        self._initialized = False
        self._step = 0

    def init_run(
        self,
        *,
        log_dir: str,
        experiment_name: str,
        group: str,
        tags: list[str],
        config: dict[str, Any],
    ) -> None:
        """Initialize the W&B run.

        Authentication is handled by the wandb SDK itself (``WANDB_API_KEY``
        env var or previously configured credentials), so unlike SwanLab no
        per-run relogin is needed.

        Args:
            log_dir: Log directory (wandb local files land here).
            experiment_name: Name of this run.
            group: Experiment group (model class name).
            tags: Tags for the run.
            config: Experiment configuration dict.
        """
        import wandb

        wandb.init(
            project=os.getenv("KT_WANDB_PROJECT") or "UniKT",
            entity=os.getenv("KT_WANDB_ENTITY") or None,
            name=experiment_name,
            dir=log_dir,
            config=config,
            group=group,
            tags=tags,
            reinit="finish_previous",
        )
        self._initialized = True

    def _commit(self, payload: dict[str, Any]) -> None:
        """Log a payload on the next self-managed monotonic step."""
        import wandb

        wandb.log(payload, step=self._step)
        self._step += 1

    @staticmethod
    def _prefix(phase: str, stage: str | None) -> str:
        """Build a W&B metric prefix from phase and optional stage."""
        return f"{stage}/{phase}/" if stage else f"{phase}/"

    def log_metrics(
        self,
        *,
        phase: str,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log epoch-level metrics to W&B.

        Metric names are grouped as ``{stage/}{phase}/{name}`` (e.g.
        ``train/acc``, ``km/val/auc``) so wandb auto-groups panels by slash.

        Args:
            phase: Phase name (e.g. ``"train"``, ``"val"``).
            metrics: Metric name to value mapping.
            step: Ignored — see class docstring (callers pass a non-monotonic
                mix of epoch- and batch-based steps).
            epoch: Current epoch number (unused in W&B).
            stage: Stage name (optional).
        """
        if not self._initialized:
            return
        prefix = self._prefix(phase, stage)
        payload = {f"{prefix}{name}": v for name, v in metrics.items() if v is not None}
        if payload:
            self._commit(payload)

    def log_early_stopping(
        self,
        *,
        phase: str,
        best_score: float | None,
        num_bad_epochs: int,
        best_metrics: dict[str, float] | None,
        step: int,
        epoch: int,
        stage: str | None = None,
    ) -> None:
        """Log early stopping trajectory to W&B.

        Args:
            phase: Phase name.
            best_score: Best monitored score.
            num_bad_epochs: Consecutive epochs without improvement.
            best_metrics: Best metric values (optional).
            step: Ignored — see class docstring.
            epoch: Current epoch number (unused in W&B).
            stage: Stage name (optional).
        """
        if not self._initialized:
            return
        sp = f"{stage}/" if stage else ""
        data = {
            f"{sp}early_stopping/best_score": best_score,
            f"{sp}early_stopping/num_bad_epochs": num_bad_epochs,
        }
        if best_metrics:
            data.update(
                {f"{sp}early_stopping/best_{k}": v for k, v in best_metrics.items()}
            )
        self._commit(data)

    def log_batch(self, **kwargs: Any) -> None:
        """No-op: W&B does not log per-batch metrics to avoid noise."""

    def log_final(self, *, metrics: dict[str, float], step: int) -> None:
        """Log final summary metrics to W&B.

        Args:
            metrics: Final metric values (keys already carry a ``Final/``
                prefix from the caller).
            step: Ignored — see class docstring.
        """
        if not self._initialized or not metrics:
            return
        self._commit(metrics)

    def log_timing(
        self,
        *,
        step: int,
        epoch: int,
        timings: dict[str, float],
        stage: str | None = None,
    ) -> None:
        """Log per-epoch timing breakdown to W&B for a stage."""
        if not self._initialized:
            return
        sp = f"{stage}/" if stage else ""
        data = {
            f"{sp}time/{k.replace('_time', '')}": v
            for k, v in timings.items()
            if v is not None
        }
        if data:
            self._commit(data)

    def finish(self) -> None:
        """Finish the W&B run and clean up."""
        if not self._initialized:
            return
        import wandb

        wandb.finish()
        self._initialized = False


class MetricLoggerComposite(MetricLogger):
    """Composite wrapper that fans out to multiple metric logger backends.

    Each method call is forwarded to every backend; exceptions from
    individual backends are caught and logged.
    """

    def __init__(self, loggers: list[MetricLogger]):
        """Initialize the composite logger.

        Args:
            loggers: List of MetricLogger instances to fan out to.
        """
        self._loggers = loggers

    def _fanout(self, method: str, **kwargs: Any) -> None:
        """Call a method on all wrapped loggers, isolating exceptions.

        ``ImportError`` propagates instead: a missing dependency fails every
        call for the rest of the run, so isolating it would silently drop
        that backend's records behind a single warning.
        """
        for lg in self._loggers:
            try:
                getattr(lg, method)(**kwargs)
            except ImportError:
                raise
            except Exception as e:
                logger.warning(f"{type(lg).__name__}.{method} failed: {e}")

    def init_run(self, **kwargs: Any) -> None:
        """Initialize all backends."""
        self._fanout("init_run", **kwargs)

    def log_metrics(self, **kwargs: Any) -> None:
        """Log metrics to all backends."""
        self._fanout("log_metrics", **kwargs)

    def log_early_stopping(self, **kwargs: Any) -> None:
        """Log early stopping to all backends."""
        self._fanout("log_early_stopping", **kwargs)

    def log_batch(self, **kwargs: Any) -> None:
        """Log batch metrics to all backends."""
        self._fanout("log_batch", **kwargs)

    def log_final(self, **kwargs: Any) -> None:
        """Log final metrics to all backends."""
        self._fanout("log_final", **kwargs)

    def log_timing(self, **kwargs: Any) -> None:
        """Log per-epoch timing breakdown to all backends."""
        self._fanout("log_timing", **kwargs)

    def finish(self) -> None:
        """Finish logging on all backends."""
        self._fanout("finish")


class AsyncMetricLoggerProxy(MetricLogger):
    """Offload metric logging to a single background thread.

    Wraps any backend and forwards every ``log_*`` call via a one-worker
    ``ThreadPoolExecutor`` so network/disk I/O (SwanLab uploads, per-row CSV
    flushes) never blocks the training thread. Mirrors the lifecycle of
    ``CheckpointManager``: submit-to-queue, ``flush`` drains tracked futures,
    idempotent ``close`` registered with ``atexit``.

    Not registered — wired directly by ``build_default_metric_loggers``, like
    ``MetricLoggerComposite``.
    """

    def __init__(self, inner: MetricLogger):
        """Initialize the proxy around ``inner``.

        Args:
            inner: The wrapped backend. Touched only by the background worker
                (plus the main thread during sync ``init_run`` and the
                post-close fallback), so no locking is needed.
        """
        self._inner = inner
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="metric-log"
        )
        self._futures: list[Future] = []
        self._closed = False
        self._creator_pid = os.getpid()
        atexit.register(self.close)

    def init_run(self, **kwargs: Any) -> None:
        """Initialize the wrapped backend synchronously.

        ``swanlab.init`` is network I/O; a failure must surface immediately
        rather than after a silent run. The composite's ``_fanout`` turns a
        non-import failure here into a warning while other backends keep
        working; ``ImportError`` propagates (see ``_fanout``).
        """
        self._inner.init_run(**kwargs)

    def _check_completed(self) -> None:
        """Drop finished futures, surfacing their exceptions as warnings.

        Bounds memory — without this, ``log_batch``-per-batch would retain
        every future until ``finish`` — and restores near-inline error
        visibility (within one submit) instead of deferring to ``flush``.
        """
        if not self._futures:
            return
        pending: list[Future] = []
        for fut in self._futures:
            if fut.done():
                try:
                    fut.result()
                except Exception as e:
                    logger.warning(f"Async metric logging failed: {e}")
            else:
                pending.append(fut)
        self._futures = pending

    def _submit(self, method: str, **kwargs: Any) -> None:
        """Forward a backend method to the worker, or run it sync after close."""
        self._check_completed()
        if self._closed:
            getattr(self._inner, method)(**kwargs)
            return
        self._futures.append(
            self._executor.submit(getattr(self._inner, method), **kwargs)
        )

    def log_metrics(self, **kwargs: Any) -> None:
        """Log epoch metrics asynchronously."""
        self._submit("log_metrics", **kwargs)

    def log_early_stopping(self, **kwargs: Any) -> None:
        """Log early stopping trajectory asynchronously."""
        self._submit("log_early_stopping", **kwargs)

    def log_batch(self, **kwargs: Any) -> None:
        """Log per-batch metrics asynchronously."""
        self._submit("log_batch", **kwargs)

    def log_final(self, **kwargs: Any) -> None:
        """Log final metrics asynchronously."""
        self._submit("log_final", **kwargs)

    def log_timing(self, **kwargs: Any) -> None:
        """Log per-epoch timing asynchronously."""
        self._submit("log_timing", **kwargs)

    def flush(self) -> None:
        """Wait for all pending log calls to finish, logging any exceptions."""
        futures, self._futures = self._futures, []
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"Async metric logging failed: {e}")

    def finish(self) -> None:
        """Drain pending logs, then finish the backend on the main thread.

        ``swanlab.finish`` restores its SIGINT handler via ``signal.signal``
        (main-thread-only) and blocks on upload confirmation that is only
        interruptible via SIGINT — both require the main thread, so finish
        must run here, not on the worker.
        """
        self.flush()
        self._inner.finish()

    def close(self) -> None:
        """Drain the queue and shut down the executor (atexit-only).

        Idempotent and fork-safe: skipped in child processes (the worker
        thread does not survive ``fork``) and never raises.
        """
        if os.getpid() != self._creator_pid or self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            with suppress(Exception):
                self._executor.shutdown(wait=True)


def get_metric_logger(name: str, **kwargs: Any) -> MetricLogger:
    """Instantiate a registered metric logger backend by name.

    Args:
        name: Backend registration name (e.g. ``"local"``, ``"swanlab"``).
        **kwargs: Arguments forwarded to the backend constructor.

    Returns:
        An instance of the requested MetricLogger subclass.
    """
    cls = METRIC_LOGGERS.get(name)
    return cls(**kwargs)


def _async_enabled() -> bool:
    """Read the ``METRIC_LOGGING_ASYNC`` env var (default enabled).

    Disabled by ``0``/``false``/``no``/``""`` (case-insensitive); any other
    value enables it. Treated as infrastructure (like ``LOG_LEVEL``), not an
    experiment parameter, so no CLI plumbing.
    """
    val = os.getenv("METRIC_LOGGING_ASYNC", "1").strip().lower()
    return val not in ("0", "false", "no", "")


def _select_tracking_backend() -> str:
    """Read the ``KT_TRACKING_BACKEND`` env var (default ``swanlab``).

    Returns ``"swanlab"`` or ``"wandb"``. An unrecognized value raises
    ``ValueError`` rather than silently rerouting to the wrong backend — a
    typo would otherwise push every run to SwanLab while the user watches
    the W&B dashboard. Treated as infrastructure (like ``LOG_LEVEL``), not
    an experiment parameter, so no CLI plumbing.
    """
    backend = os.getenv("KT_TRACKING_BACKEND", "swanlab").strip().lower() or "swanlab"
    if backend not in ("swanlab", "wandb"):
        raise ValueError(
            f"Unknown KT_TRACKING_BACKEND={backend!r}. Valid: swanlab | wandb."
        )
    return backend


def build_default_metric_loggers(
    *,
    log_dir: str,
    log_batch_metrics: bool,
    cloud_tracking: bool,
    async_io: bool | None = None,
) -> MetricLogger:
    """Build the default metric logger composite.

    Local CSV logging is always enabled. When ``cloud_tracking`` is ``True``,
    exactly one remote backend is added, selected by ``KT_TRACKING_BACKEND``
    (``swanlab`` | ``wandb``, default ``swanlab``) — the two are mutually
    exclusive, and an unimportable backend raises ``RuntimeError`` here
    rather than degrading to a warning once training has started. Each
    backend is wrapped in :class:`AsyncMetricLoggerProxy`
    when async I/O is enabled (default), so a slow remote upload cannot
    serialize the local CSV write.

    Args:
        log_dir: Directory for local CSV logs.
        log_batch_metrics: Whether to log per-batch loss.
        cloud_tracking: If False, skip the remote backend entirely.
        async_io: Override async I/O. ``None`` reads ``METRIC_LOGGING_ASYNC``
            (default enabled).

    Returns:
        A MetricLoggerComposite instance.
    """
    if async_io is None:
        async_io = _async_enabled()

    loggers: list[MetricLogger] = [
        get_metric_logger("local", log_dir=log_dir, log_batch_metrics=log_batch_metrics)
    ]
    if cloud_tracking:
        backend = _select_tracking_backend()
        try:
            importlib.import_module(backend)
        except ImportError as e:
            raise RuntimeError(
                f"Cloud tracking backend '{backend}' is not importable: {e}. "
                f"Install it in this environment, or disable cloud tracking "
                f"with --general.cloud_tracking false."
            ) from e
        loggers.append(get_metric_logger(backend))

    if async_io:
        logger.debug(
            "Async metric logging: ENABLED (METRIC_LOGGING_ASYNC=0 to disable)"
        )
        loggers = [AsyncMetricLoggerProxy(lg) for lg in loggers]
    return MetricLoggerComposite(loggers)


__all__ = [
    "AsyncMetricLoggerProxy",
    "LocalMetricLogger",
    "MetricLogger",
    "MetricLoggerComposite",
    "SwanLabMetricLogger",
    "WandbMetricLogger",
    "build_default_metric_loggers",
    "get_metric_logger",
]
