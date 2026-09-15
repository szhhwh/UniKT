"""Training callback system.

Provides a callback mechanism for the training process, including
early stopping, checkpointing, memory management, and test evaluation.
"""

from abc import ABC
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

from rich.console import Group
from rich.live import Live
from rich.text import Text

from ..core import get_logger
from ..progress import create_progress
from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping

logger = get_logger(__name__)

TCallback = TypeVar("TCallback", bound="Callback")


class Callback(ABC):
    """Base class for training callbacks.

    Defines the callback interface for the training loop. Subclasses
    can implement specific callback logic for hooks at various points
    during training.
    """

    def on_train_begin(self, epochs: int, **kwargs):
        """Called when training begins.

        Args:
            epochs: Total number of epochs.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def on_train_end(self, **kwargs):
        """Called when training ends.

        Args:
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Called when an epoch begins.

        Args:
            epoch: Current epoch number.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Called when an epoch ends.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch (None if no val data).
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def on_phase_begin(self, epoch: int, phase: str, **kwargs):
        """Called when a training/validation phase begins.

        Args:
            epoch: Current epoch number.
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Called when a training/validation phase ends.

        Args:
            epoch: Current epoch number.
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
            loss: Mean loss per batch for this phase.
            metrics: Metric dictionary for this phase.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str, **kwargs):
        """Called when a batch begins.

        Args:
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def on_batch_end(
        self, epoch: int, batch_idx: int, phase: str, loss: float, **kwargs
    ):
        """Called when a batch ends.

        Args:
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
            loss: Loss value for this batch.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        pass

    def should_stop(self, **kwargs) -> bool:
        """Check whether training should stop.

        Args:
            **kwargs: Additional keyword arguments (e.g. trainer).

        Returns:
            True if training should stop, False otherwise.
        """
        return False

    def close(self):
        """Release resources held by this callback.

        Called from the trainer's finally path — including failed runs —
        unlike ``on_train_end`` which only fires on normal completion.
        Implementations must be idempotent and must not raise.
        """
        pass


class FunctionCallback(Callback):
    """Functional callback wrapper.

    Maps event names to callable functions via a dictionary for
    lightweight callback registration without subclassing.
    """

    def __init__(self, handlers: dict[str, Callable | list[Callable]]):
        """Initialize the function callback wrapper.

        Args:
            handlers: Dictionary mapping event names to callables
                or lists of callables.
        """
        self._handlers: dict[str, list[Callable]] = {}
        for name, funcs in handlers.items():
            if funcs is None:
                continue
            if isinstance(funcs, list):
                self._handlers[name] = funcs
            else:
                self._handlers[name] = [funcs]

    def _call(self, name: str, *args, **kwargs) -> None:
        """Invoke all handlers registered for a given event name."""
        for func in self._handlers.get(name, []):
            func(*args, **kwargs)

    def on_train_begin(self, epochs: int, **kwargs):
        """Delegate to registered ``on_train_begin`` handlers."""
        self._call("on_train_begin", epochs)

    def on_train_end(self, **kwargs):
        """Delegate to registered ``on_train_end`` handlers."""
        self._call("on_train_end")

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Delegate to registered ``on_epoch_begin`` handlers."""
        self._call("on_epoch_begin", epoch)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Delegate to registered ``on_epoch_end`` handlers."""
        self._call("on_epoch_end", epoch, train_loss, val_loss)

    def on_phase_begin(self, epoch: int, phase: str, **kwargs):
        """Delegate to registered ``on_phase_begin`` handlers."""
        self._call("on_phase_begin", epoch, phase)

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Delegate to registered ``on_phase_end`` handlers."""
        self._call("on_phase_end", epoch, phase, loss, metrics)

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str, **kwargs):
        """Delegate to registered ``on_batch_begin`` handlers."""
        self._call("on_batch_begin", epoch, batch_idx, phase)

    def on_batch_end(
        self, epoch: int, batch_idx: int, phase: str, loss: float, **kwargs
    ):
        """Delegate to registered ``on_batch_end`` handlers."""
        self._call("on_batch_end", epoch, batch_idx, phase, loss)

    def should_stop(self, **kwargs) -> bool:
        """Check all registered ``should_stop`` handlers.

        Returns:
            True if any handler returns a truthy value.
        """
        results = [func() for func in self._handlers.get("should_stop", [])]
        return any(bool(result) for result in results)


class CallbackManager:
    """Manages a list of callback objects.

    Provides methods to trigger lifecycle events on all registered
    callbacks in order and to query whether training should stop.
    """

    def __init__(self, callbacks: list[Callback]):
        """Initialize the callback manager.

        Args:
            callbacks: List of callback instances.
        """
        self.callbacks = [cb for cb in callbacks if cb is not None]

    def trigger(self, method_name: str, *args, **kwargs):
        """Invoke a method on all registered callbacks.

        Args:
            method_name: Name of the callback method to invoke.
            *args: Positional arguments forwarded to each callback.
            **kwargs: Keyword arguments forwarded to each callback.
        """
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

    def get_callback(self, callback_type: type[TCallback]) -> TCallback | None:
        """Get the first callback of a specified type.

        Args:
            callback_type: The callback class to search for.

        Returns:
            The first matching callback, or None if not found.
        """
        for cb in self.callbacks:
            if isinstance(cb, callback_type):
                return cb
        return None

    def should_stop(self, **kwargs) -> bool:
        """Check whether any callback requests training to stop.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            True if any callback's ``should_stop`` returns True.
        """
        return any(cb.should_stop(**kwargs) for cb in self.callbacks)

    def close(self):
        """Call ``close`` on all callbacks in list order.

        Runs on the trainer's finally path (including failed runs), so a
        raising ``close`` must not mask the original training error or
        skip the callbacks after it; each call is isolated here.
        Implementations must still be idempotent and must not raise.
        """
        for callback in self.callbacks:
            try:
                callback.close()
            except Exception as e:
                logger.warning(f"{type(callback).__name__}.close() failed: {e}")

    # Convenience methods
    def on_train_begin(self, epochs: int, **kwargs):
        """Trigger ``on_train_begin`` on all callbacks.

        Args:
            epochs: Total number of epochs.
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_train_begin", epochs, **kwargs)

    def on_train_end(self, **kwargs):
        """Trigger ``on_train_end`` on all callbacks.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_train_end", **kwargs)

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Trigger ``on_epoch_begin`` on all callbacks.

        Args:
            epoch: Current epoch number.
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_epoch_begin", epoch, **kwargs)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Trigger ``on_epoch_end`` on all callbacks.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_epoch_end", epoch, train_loss, val_loss, **kwargs)

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Trigger ``on_phase_end`` on all callbacks.

        Args:
            epoch: Current epoch number.
            phase: Phase name.
            loss: Mean loss per batch for this phase.
            metrics: Metric dictionary for this phase.
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_phase_end", epoch, phase, loss, metrics, **kwargs)

    def on_phase_begin(self, epoch: int, phase: str, **kwargs):
        """Trigger ``on_phase_begin`` on all callbacks.

        Args:
            epoch: Current epoch number.
            phase: Phase name.
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_phase_begin", epoch, phase, **kwargs)

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str, **kwargs):
        """Trigger ``on_batch_begin`` on all callbacks.

        Args:
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            phase: Phase name.
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_batch_begin", epoch, batch_idx, phase, **kwargs)

    def on_batch_end(
        self, epoch: int, batch_idx: int, phase: str, loss: float, **kwargs
    ):
        """Trigger ``on_batch_end`` on all callbacks.

        Args:
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            phase: Phase name.
            loss: Loss value for this batch.
            **kwargs: Additional keyword arguments.
        """
        self.trigger("on_batch_end", epoch, batch_idx, phase, loss, **kwargs)


class EarlyStoppingCallback(Callback):
    """Early stopping callback.

    Monitors a validation metric and triggers early stopping when the
    metric stops improving.
    """

    def __init__(
        self,
        *,
        early_stopping: EarlyStopping,
        stage: str | None = None,
    ):
        """Initialize the early stopping callback.

        Args:
            early_stopping: Early stopping controller instance.
            stage: Stage name for multi-stage training (None for single-stage).
        """
        self.early_stopping = early_stopping
        self.cfg = early_stopping.cfg
        self.stage = stage

        self._stop = False

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Run early stopping check at the end of the validation phase.

        Args:
            epoch: Current epoch number.
            phase: Phase name (only ``"val"`` triggers the check).
            loss: Mean loss per batch for this phase.
            metrics: Metric dictionary for this phase.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        if phase != "val":
            return
        current = self._select_monitor_value(metrics, loss)
        self.step(current, epoch, metrics)
        trainer = kwargs.get("trainer")
        metric_logger = getattr(trainer, "metric_logger", None) if trainer else None
        if metric_logger is not None:
            metric_logger.log_early_stopping(
                phase="val",
                best_score=self.early_stopping.best_score,
                num_bad_epochs=self.early_stopping.num_bad_epochs,
                best_metrics=self.early_stopping.best_metrics,
                step=getattr(trainer, "_global_step", epoch),
                epoch=epoch,
                stage=self.stage,
            )

    def step(self, current: float, epoch: int, metrics: dict | None = None) -> bool:
        """Execute the early stopping check.

        Args:
            current: Current metric value.
            epoch: Current epoch number.
            metrics: Full metrics dictionary (optional).

        Returns:
            True if training should stop, False otherwise.
        """
        self._stop = self.early_stopping.step(current, epoch, metrics)
        return self._stop

    def _select_monitor_value(self, metrics: dict, val_loss: float | None) -> float:
        """Select the monitored value from metrics or loss, with fallbacks."""
        name = (self.cfg.monitor or "auc").lower()
        value = None
        if name == "loss":
            value = float(val_loss) if val_loss is not None else None
        elif name in metrics:
            value = metrics[name]

        if value is None:
            if metrics.get("auc") is not None:
                value = float(metrics["auc"])
            elif metrics.get("auprc") is not None:
                value = float(metrics["auprc"])
            elif metrics.get("acc") is not None:
                value = float(metrics["acc"])
            elif metrics.get("rmse") is not None:
                value = float(metrics["rmse"])

        if value is None:
            if name in ["loss", "rmse"]:
                return float("inf")
            return float("-inf")
        return float(value)

    def should_stop(self, **kwargs) -> bool:
        """Check whether early stopping has been triggered.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            True if training should stop.
        """
        return self._stop


class CheckpointCallback(Callback):
    """Checkpoint saving callback.

    Saves model weights at the end of each validation phase when a
    monitored metric improves, and saves a full checkpoint at the end
    of each epoch.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        *,
        early_stopping: EarlyStopping | None = None,
        last_filename: str = "last_checkpoint.pth",
        best_filename: str | None = "best_model.pth",
        keep_best_state: bool = True,
        save_last_checkpoint: bool = True,
        monitor: str | None = None,
        mode: str | None = None,
    ):
        """Initialize the checkpoint callback.

        Args:
            checkpoint_manager: Checkpoint manager instance.
            early_stopping: Early stopping controller (optional).
            last_filename: Filename for the last checkpoint.
            best_filename: Filename for the best model (None to skip).
            keep_best_state: Whether to cache the best model state_dict
                in memory (useful for multi-stage training).
            save_last_checkpoint: Whether to save the full last checkpoint
                at the end of every epoch.
            monitor: Metric name to monitor for best model selection.
                Defaults to the early stopping monitor if set; overrides it
                when explicitly provided to decouple checkpointing from
                early stopping.
            mode: Direction for the monitored metric (``"max"`` or ``"min"``).
                Inferred from the monitor or early stopping configuration.
        """
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.last_filename = last_filename
        self.best_filename = best_filename
        self.keep_best_state = keep_best_state
        self.save_last_checkpoint = save_last_checkpoint
        self._monitor_override = monitor.lower() if monitor else None
        self._mode_override = mode.lower() if mode else None

        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.best_model_state: dict | None = None

    def on_train_begin(self, epochs: int, **kwargs):
        """Reset tracking state at the start of training.

        Args:
            epochs: Total number of epochs.
            **kwargs: Additional keyword arguments.
        """
        trainer = kwargs.get("trainer")
        if trainer is not None and getattr(trainer, "_resumed", False):
            return
        self.best_metric = None
        self.best_epoch = None
        self.best_model_state = None

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Save the best model at the end of the validation phase.

        Args:
            epoch: Current epoch number.
            phase: Phase name (only ``"val"`` triggers saving).
            loss: Mean loss per batch for this phase.
            metrics: Metric dictionary for this phase.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        trainer = kwargs.get("trainer")
        if trainer is None or phase != "val" or self.best_filename is None:
            return

        current = self._select_monitor_value(metrics, loss)
        if not self._is_better_metric(current):
            return

        self.best_metric = current
        self.best_epoch = epoch
        snapshot = self.checkpoint_manager.save_weights(
            trainer.model, self.best_filename
        )
        self.best_model_state = snapshot if self.keep_best_state else None

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Save the last checkpoint at the end of each epoch.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        if not self.save_last_checkpoint:
            return
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        self.checkpoint_manager.save_checkpoint(
            epoch,
            trainer.model,
            trainer.opt,
            trainer.lr_scheduler,
            early_stopping_state=self._get_early_stopping_state(),
            filename=self.last_filename,
        )

    def _monitor_name(self) -> str:
        """Get the metric name to monitor for best model selection."""
        if self._monitor_override:
            return self._monitor_override
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _select_monitor_value(self, metrics: dict, val_loss: float | None) -> float:
        """Select the current monitored value from metrics or loss.

        Falls back to auc → auprc → acc → rmse if the named metric is unavailable.
        """
        name = self._monitor_name()
        value = None
        if name == "loss":
            value = float(val_loss) if val_loss is not None else None
        elif name in metrics:
            value = metrics[name]

        if value is None:
            if metrics.get("auc") is not None:
                value = float(metrics["auc"])
            elif metrics.get("auprc") is not None:
                value = float(metrics["auprc"])
            elif metrics.get("acc") is not None:
                value = float(metrics["acc"])
            elif metrics.get("rmse") is not None:
                value = float(metrics["rmse"])

        if value is None:
            if name in ["loss", "rmse"]:
                return float("inf")
            return float("-inf")
        return float(value)

    def _is_better_metric(self, current: float) -> bool:
        """Determine whether the current metric is better than the best so far.

        Respects the mode (max/min) from explicit config, early stopping,
        or metric name convention.
        """
        if self.best_metric is None:
            return True
        if self._mode_override:
            mode = self._mode_override
        elif self.early_stopping is not None:
            mode = self.early_stopping.cfg.mode
        elif self._monitor_name() in ["rmse", "loss"]:
            mode = "min"
        else:
            mode = "max"
        return (
            current > self.best_metric if mode == "max" else current < self.best_metric
        )

    def _get_early_stopping_state(self) -> dict | None:
        """Serialize the early stopping state for checkpoint persistence."""
        if self.early_stopping is None:
            return None
        state = {
            "best_score": self.early_stopping.best_score,
            "best_epoch": self.early_stopping.best_epoch,
            "num_bad_epochs": self.early_stopping.num_bad_epochs,
        }
        if self.early_stopping.best_metrics is not None:
            state["best_metrics"] = self.early_stopping.best_metrics.copy()
        return state


class MemoryCleanupCallback(Callback):
    """Callback for periodic GPU memory cleanup and garbage collection.

    Calls ``torch.cuda.empty_cache()`` and Python ``gc.collect()`` at
    regular intervals during training.
    """

    def __init__(self, cleanup_interval: int = 5):
        """Initialize the memory cleanup callback.

        Args:
            cleanup_interval: Interval in epochs between cleanup runs.
        """
        self.cleanup_interval = cleanup_interval

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Run cleanup at the end of a phase if the interval is met.

        Args:
            epoch: Current epoch number.
            phase: Phase name.
            loss: Mean loss per batch for this phase.
            metrics: Metric dictionary for this phase.
            **kwargs: Additional keyword arguments.
        """
        if epoch % self.cleanup_interval == 0:
            self._cleanup_memory(phase)

    def _cleanup_memory(self, phase: str):
        """Clean up GPU cache and run Python garbage collection.

        Logs a debug message if more than 100 MB of GPU memory was
        reclaimed.
        """
        import gc

        import torch

        if torch.cuda.is_available():
            try:
                before = torch.cuda.memory_allocated() / 1024**3  # GB
                torch.cuda.empty_cache()
                after = torch.cuda.memory_allocated() / 1024**3
                if before - after > 0.1:  # > 100 MB recovered
                    logger.debug(
                        f"[{phase}] Cleaned up memory: {before:.2f}GB -> {after:.2f}GB"
                    )
            except Exception as e:
                logger.warning(f"Memory cleanup failed: {e}")

        # Python garbage collection
        gc.collect()


class TestEvaluationCallback(Callback):
    """Callback that runs test set evaluation at the end of training."""

    def __init__(self, *, use_best_model: bool = True):
        """Initialize the test evaluation callback.

        Args:
            use_best_model: Whether to load the best model before evaluation.
        """
        self.use_best_model = use_best_model

    def on_train_end(self, **kwargs):
        """Run test set evaluation when training ends.

        Args:
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        trainer._evaluate_on_test_set(use_best_model=self.use_best_model)


class ProgressCallback(Callback):
    """Rich progress rendering for the training loop.

    Owns all live rendering (epoch bar, per-phase batch bar, best-metric
    header) so ``BaseTrainer._run_training_loop`` stays pure orchestration.
    Reads every value from the ``trainer`` kwarg at hook time. Trainers
    build a fresh instance per run/stage (see ``_maybe_progress_callback``);
    ``on_train_begin`` still tears down any previous display, so an
    explicitly shared instance also survives re-entry across stages.
    """

    # rich allows one live display per console: the latest instance takes
    # over from an older one so duplicate registrations degrade to a single
    # display instead of raising LiveError at the second start.
    _active: ClassVar["ProgressCallback | None"] = None

    def __init__(self):
        """Initialize with no display; everything is built in ``on_train_begin``."""
        self._trainer: Any = None
        self._progress = None
        self._live: Live | None = None
        self._best_text: Text | None = None
        self._total_task = None
        self._work_task = None
        self._checkpoint_cb: CheckpointCallback | None = None
        self._monitor_name = "auc"
        self._stage_prefix = ""

    def on_train_begin(self, epochs: int, **kwargs):
        """Build the live display for the upcoming (stage) run.

        Args:
            epochs: Total number of epochs.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        self._teardown()
        other = ProgressCallback._active
        if other is not None and other is not self:
            other._teardown()
        ProgressCallback._active = self

        self._trainer = trainer
        self._checkpoint_cb = trainer.callback_manager.get_callback(CheckpointCallback)
        self._monitor_name = (
            self._checkpoint_cb._monitor_name()
            if self._checkpoint_cb is not None
            else trainer._monitor_name()
        )
        self._stage_prefix = (
            f"[{trainer._current_stage.upper()}] " if trainer._current_stage else ""
        )
        total_label = (
            f"{self._stage_prefix}Epochs" if trainer._current_stage else "Total Epochs"
        )

        progress = create_progress()
        renderables = [progress]
        if trainer.early_stopping is not None:
            self._best_text = Text(
                f"{self._stage_prefix}Best {self._monitor_name.upper()}: N/A",
                style="bold yellow",
            )
            renderables.insert(0, self._best_text)

        # Manual start/stop: the display spans hook calls, so a with-block
        # owned by this method cannot scope it.
        self._progress = progress
        self._live = Live(Group(*renderables))
        self._live.start()
        self._total_task = progress.add_task(
            f"[bold red]{total_label}",
            total=trainer.epochs,
            completed=trainer.start_epoch,
        )
        self._work_task = progress.add_task(
            "[bold green]Training", total=len(trainer.train_data)
        )

    def on_phase_begin(self, epoch: int, phase: str, **kwargs):
        """Reset the batch bar for the starting phase.

        Args:
            epoch: Current epoch number.
            phase: Phase name, ``"train"`` or ``"val"``.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        if self._progress is None or self._trainer is None:
            return
        loader = (
            self._trainer.train_data if phase == "train" else self._trainer.val_data
        )
        description = (
            "[bold green]Training" if phase == "train" else "[bold cyan]Validation"
        )
        self._progress.reset(
            self._work_task, total=len(loader), description=description
        )

    def on_batch_end(
        self, epoch: int, batch_idx: int, phase: str, loss: float, **kwargs
    ):
        """Advance the batch bar by one batch.

        Args:
            epoch: Current epoch number.
            batch_idx: Batch index within the epoch.
            phase: Phase name, ``"train"`` or ``"val"``.
            loss: Loss value for this batch.
            **kwargs: Additional keyword arguments.
        """
        if self._progress is not None:
            self._progress.advance(self._work_task)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Advance the epoch bar and refresh the best-metric header.

        Best metric values are already updated by ``CheckpointCallback`` at
        ``on_phase_end`` (all callbacks fire there before any
        ``on_epoch_end``), so they are fresh regardless of list order.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_loss: Validation loss for this epoch.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        if self._progress is None or self._trainer is None:
            return
        self._progress.advance(self._total_task)

        trainer = self._trainer
        if (
            self._best_text is not None
            and trainer.val_data is not None
            and trainer.early_stopping is not None
        ):
            best_metric = (
                self._checkpoint_cb.best_metric
                if self._checkpoint_cb is not None
                else None
            )
            best_epoch = (
                self._checkpoint_cb.best_epoch
                if self._checkpoint_cb is not None
                else None
            )
            patience = trainer.early_stopping.cfg.patience
            remaining = max(0, patience - trainer.early_stopping.num_bad_epochs)
            best_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
            self._best_text.plain = (
                f"{self._stage_prefix}Best {self._monitor_name.upper()}: {best_str} "
                f"(Epoch {best_epoch + 1 if best_epoch is not None else 'N/A'}, "
                f"Patience: {remaining}/{patience})"
            )
            self._best_text.stylize("bold yellow")

    def on_train_end(self, **kwargs):
        """Stop the live display; later callbacks (test evaluation) render freely.

        Args:
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        self._teardown()

    def close(self):
        """Stop the live display; idempotent and safe on failed runs."""
        self._teardown()

    def _teardown(self):
        """Stop and drop the display; must never mask a training error."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception as e:
                logger.debug(f"Progress display teardown failed: {e}")
        self._live = None
        self._progress = None
        self._best_text = None
        self._total_task = None
        self._work_task = None
        self._checkpoint_cb = None
        self._trainer = None
        if ProgressCallback._active is self:
            ProgressCallback._active = None


__all__ = [
    "Callback",
    "CallbackManager",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "FunctionCallback",
    "MemoryCleanupCallback",
    "ProgressCallback",
    "TestEvaluationCallback",
]
