"""Base trainer module.

Provides the core trainer functionality including device management,
data loading, training loops, callbacks, checkpointing, and metric
logging.
"""

import datetime
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ..config import create_optimized_dataloader
from ..core import get_logger
from ..progress import create_progress, resolve_progress
from .callbacks import (
    Callback,
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    MemoryCleanupCallback,
    ProgressCallback,
    TestEvaluationCallback,
)
from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping
from .inference_ops import InferenceOpsMixin
from .metric_logger import build_default_metric_loggers
from .metrics import MetricsAccumulator
from .runtime_components import RuntimeComponents

logger = get_logger(__name__)


@dataclass
class StageResult:
    """Result of a single training stage.

    Attributes:
        name: Stage name (``None`` for single-stage training).
        best_metric: Best monitored metric value on the validation set.
        best_epoch: Epoch (0-indexed) at which the best metric was achieved.
        final_epoch: Last epoch actually trained in this stage.
        monitor: Name of the monitored metric (e.g. ``"auc"``, ``"acc"``,
            ``"rmse"``).
    """

    name: str | None = None
    best_metric: float | None = None
    best_epoch: int | None = None
    final_epoch: int | None = None
    monitor: str = "auc"


class BaseTrainer(InferenceOpsMixin, ABC):
    r"""Abstract base class for trainers.

    A single-stage trainer is constructed in one step::

        trainer = MyTrainer(rc, data_src, exp_manager)
        trainer.run()

    The constructor wires everything: it calls :meth:`build_components` to let
    the subclass assemble the model/optimizer/data from ``rc`` + ``data_src``,
    reads :meth:`build_callbacks` for any extra callbacks, then runs
    :meth:`build` to finalize device, loaders, early stopping, logging and
    checkpointing.

    Subclasses implement:

    1. :meth:`build_components`: Return a :class:`RuntimeComponents` holding the
       model, optimizer, loss, scheduler, and train/val/test data built from
       ``rc`` + ``data_src``. Scalar knobs are read from ``rc`` directly, never
       stored on the components.
    2. :meth:`build_callbacks`: Return extra callbacks (default ``[]``).
    3. :meth:`forward_pass`: Model forward pass for one batch.
    4. :meth:`_compute_loss`: Training loss (default ``self.loss(y_hat,
       y_label)``); auxiliary terms added here are excluded from val/test
       loss, which always goes through :meth:`_compute_eval_loss`.
    """

    def __init__(self, rc: Any, data_src: Any, exp_manager: Any = None) -> None:
        """Construct and build the trainer in one step.

        Args:
            rc: RunConfig instance — the single source of
                truth for scalar configuration.
            data_src: Data source used by :meth:`build_components` to prepare
                datasets and model metadata.
            exp_manager: Experiment manager (run directory / tracking). May be
                ``None`` only for inference-only subclasses that override the
                constructor.
        """
        self._init_trainer_state(rc, data_src, exp_manager)
        self._components = self.build_components(rc, data_src)
        self._custom_callbacks: list[Callback] = self.build_callbacks()
        self.build()

    def _init_trainer_state(self, rc: Any, data_src: Any, exp_manager: Any) -> None:
        """Initialize shared trainer instance state (no build side effects).

        Called by :meth:`__init__` and by :class:`MultiTrainer`, which builds
        per-stage components instead of a single ``build_components``.
        """
        self.run_config = rc
        self._data_src = data_src
        self._exp_manager = exp_manager
        # rc=None must keep failing in build() ("run_config is required"),
        # not here.
        self._progress_enabled = (
            resolve_progress(rc.general.progress) if rc is not None else False
        )
        self._components = RuntimeComponents()
        self._custom_callbacks: list[Callback] = []

        self._built = False
        self.model = None
        self.device_: torch.device | None = None
        self.epochs: int | None = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.opt = None
        self.max_clip_grad_norm: float | None = None
        self.loss = None
        self.lr_scheduler = None
        self.early_stopping: EarlyStopping | None = None
        self.start_epoch = 0
        self.log_dir = None
        self.metrics_accumulator = None
        self.checkpoint_manager = None
        self.callback_manager = None
        self.metric_logger = None
        self._global_step = 0
        self._resumed = False

        # Multi-stage context (None/0 for single-stage)
        self._current_stage: str | None = None
        self._metric_step_offset: int = 0

        # Timing statistics
        self._run_start_time: float | None = None
        self._train_end_time: float | None = None
        self._epoch_times: list[float] = []

    # ==================== Subclass hooks ====================

    def build_components(self, rc: Any, data_src: Any) -> RuntimeComponents:
        """Assemble the model/optimizer/data for this run.

        Single-stage trainers override this to return a populated
        :class:`RuntimeComponents`. The default returns an empty holder and is
        only meant for subclasses that manage their own lifecycle (e.g.
        multi-stage trainers that build per stage, and inference analyzers).
        """
        return RuntimeComponents()

    def build_callbacks(self) -> list[Callback]:
        """Return extra callbacks for this run (default: none)."""
        return []

    def add_callback(self, callback: Callback) -> None:
        """Append a callback to the active run after construction.

        Single-stage trainers already have a live callback list (built in
        :meth:`build`). Multi-stage trainers build their per-stage callback
        manager lazily from ``_custom_callbacks`` — registering there ensures the
        callback runs in every stage.
        """
        if self.callback_manager is not None:
            self.callback_manager.callbacks.append(callback)
        else:
            self._custom_callbacks.append(callback)

    # ==================== Build ====================

    def _maybe_progress_callback(self) -> list[Callback]:
        """Return the progress callback when rendering is enabled.

        Shared by single-stage :meth:`build` and the per-stage callback
        assembly in :class:`MultiTrainer`.
        """
        return [ProgressCallback()] if self._progress_enabled else []

    def build(self) -> "BaseTrainer":
        """Finalize the trainer: device, loaders, early stopping, logging.

        Reads scalar configuration from ``self.run_config`` and runtime objects
        from ``self._components`` / ``self._exp_manager``.

        Returns:
            Self for method chaining.
        """
        if self._built:
            logger.warning("Trainer already built. Skipping rebuild.")
            return self

        rc = self.run_config
        if rc is None:
            raise ValueError("run_config is required.")
        if self._exp_manager is None:
            raise ValueError("exp_manager is required.")
        c = self._components

        # 1. Device
        dev = rc.general.device
        self.device_ = torch.device(dev) if dev else self._try_gpu()

        # 2. Scalar snapshot. ``epochs`` is a per-run scalar read from rc.model;
        #    multi-stage trainers override it per stage via _apply_stage.
        self.epochs = rc.model.epochs

        # 3. Runtime instances
        self.model = c.model
        self.opt = c.optimizer
        self.loss = c.loss_fn
        self.lr_scheduler = c.lr_scheduler
        self.max_clip_grad_norm = c.max_clip_grad_norm

        # 4. Data loaders
        self._setup_data_loaders()

        # 5. Early stopping (config lives on rc)
        self.early_stopping = EarlyStopping(rc.early_stopping)

        # 6. Log directory
        exp_manager = self._exp_manager
        self.log_dir = exp_manager.get_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)

        # 7. Shared components
        self.metrics_accumulator = MetricsAccumulator()
        self.checkpoint_manager = CheckpointManager(self.log_dir)
        self.metric_logger = build_default_metric_loggers(
            log_dir=self.log_dir,
            log_batch_metrics=self.run_config.general.log_batch_metrics,
            cloud_tracking=self.run_config.general.cloud_tracking,
        )

        # 8. Callbacks
        callbacks: list[Callback] = list(self._custom_callbacks)
        callbacks.append(MemoryCleanupCallback(cleanup_interval=5))
        callbacks.append(
            EarlyStoppingCallback(early_stopping=self.early_stopping, stage=None)
        )
        callbacks.append(
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                early_stopping=self.early_stopping,
                last_filename="last_checkpoint.pth",
                best_filename="best_model.pth",
                save_last_checkpoint=self.run_config.general.save_last_checkpoint,
            )
        )
        # Before TestEvaluationCallback: its on_train_end renders outside the
        # live display, which ProgressCallback.on_train_end stops first.
        callbacks.extend(self._maybe_progress_callback())
        if not self.run_config.general.skip_test:
            callbacks.append(TestEvaluationCallback(use_best_model=True))
            if self.test_data is None:
                logger.warning(
                    "Test data was not provided. Test evaluation will be skipped. "
                    "Ensure build_components returns test_data, or set --skip_test."
                )
            else:
                test_len = (
                    len(self.test_data) if hasattr(self.test_data, "__len__") else None
                )
                if test_len is not None and test_len == 0:
                    logger.warning(
                        "Test set is empty (0 samples). Test evaluation will "
                        "produce no results. Re-run data preprocessing with a "
                        "larger test_ratio, or use --skip_test."
                    )
        else:
            logger.info("Test evaluation will be skipped.")
        self.callback_manager = CallbackManager(callbacks)

        # 9. Save RunConfig archive (skip when reusing an existing run dir, so
        #    the training archive is preserved for evaluate/case_analysis).
        if not getattr(exp_manager, "is_existing_run", False):
            self._setup_run_config_archive()

        # 10. Resume from checkpoint
        if rc.general.checkpoint_path:
            self._load_checkpoint(rc.general.checkpoint_path)

        # 11. torch.compile
        self._apply_compile()

        self._built = True
        logger.info("Trainer built successfully")
        return self

    def _setup_data_loaders(self):
        """Wrap Dataset instances into optimized DataLoaders.

        DataLoaders pass through unchanged. ``collate_fn`` applies to train; the
        val/test collators fall back to it when not set explicitly.
        """
        c = self._components
        batch_size = self.run_config.model.batch_size
        val_collate_fn = (
            c.val_collate_fn if c.val_collate_fn is not None else c.collate_fn
        )
        test_collate_fn = (
            c.test_collate_fn if c.test_collate_fn is not None else c.collate_fn
        )

        def _build_loader(data, shuffle, loader_collate_fn):
            if isinstance(data, torch.utils.data.Dataset):
                return create_optimized_dataloader(
                    data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    device=self.device_,
                    collate_fn=loader_collate_fn,
                    pin_memory=self.run_config.general.pin_memory,
                )
            return data

        self.train_data = _build_loader(c.train_data, True, c.collate_fn)
        self.val_data = _build_loader(c.val_data, False, val_collate_fn)
        self.test_data = _build_loader(c.test_data, False, test_collate_fn)

    def _setup_run_config_archive(self):
        """Save the RunConfig yaml archive plus a runtime-metadata sidecar."""
        from utils.config import save_run_config_archive

        rc = self.run_config
        metadata: dict = {
            "model_name": rc.experiment.model_name,
            "dataset_name": rc.data.dataset,
            "seed": rc.general.seed,
        }
        if self.model is not None:
            metadata["total_params"] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        if self.opt is not None:
            metadata["optimizer"] = type(self.opt).__name__
        if self.loss is not None:
            metadata["loss_function"] = type(self.loss).__name__
        if self.lr_scheduler is not None:
            metadata["lr_scheduler"] = type(self.lr_scheduler).__name__
        if hasattr(self.opt, "defaults") and "weight_decay" in self.opt.defaults:
            metadata["weight_decay"] = self.opt.defaults["weight_decay"]
        if self.device_ is not None:
            for key, value in self._get_device_info().items():
                metadata[key] = value
        save_run_config_archive(rc, self.log_dir, metadata=metadata)
        logger.info("RunConfig archive saved to %s/run_config.yaml", self.log_dir)

    def _apply_compile(self):
        """Apply ``torch.compile`` to the model when ``rc.compile`` enables it."""
        cc = self.run_config.compile
        if not cc.compile:
            return
        logger.info(
            f"Applying torch.compile: mode={cc.compile_mode}, "
            f"backend={cc.compile_backend}, "
            f"fullgraph={cc.compile_fullgraph}, dynamic={cc.compile_dynamic}"
        )
        self.model = torch.compile(
            self.model,
            mode=cc.compile_mode,
            fullgraph=cc.compile_fullgraph,
            dynamic=cc.compile_dynamic,
            backend=cc.compile_backend,
        )
        logger.info("torch.compile applied successfully")

    @abstractmethod
    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """Perform a forward pass for a single batch.

        Args:
            batch_data: A batch of data from the DataLoader.

        Returns:
            Dict containing at least ``"y_hat"``, ``"y_label"``,
            ``"y_predict"``.
        """
        raise NotImplementedError("Subclasses must implement forward_pass method")

    def test_forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """Perform a forward pass for test data.

        Defaults to :meth:`forward_pass`; override for test-specific logic
        (e.g. multi-stage where test uses a specific sub-model).

        Args:
            batch_data: A batch of test data from the DataLoader.

        Returns:
            Dict containing at least ``"y_hat"``, ``"y_label"``,
            ``"y_predict"``.
        """
        return self.forward_pass(batch_data)

    def _get_device_info(self):
        """Get device information including CUDA device details.

        Returns:
            Dict with keys like ``cuda_available``, ``cuda_device_count``,
            ``cuda_device_name``, etc.
        """
        device_info = {}

        if self.device_.type == "cuda":
            device_info["cuda_available"] = True
            device_info["cuda_device_count"] = torch.cuda.device_count()
            device_index = self.device_.index if self.device_.index is not None else 0
            device_info["cuda_device_name"] = torch.cuda.get_device_name(device_index)
            device_info["cuda_device_capability"] = torch.cuda.get_device_capability(
                device_index
            )
        else:
            device_info["cuda_available"] = False
            device_info["device_type"] = "CPU"

        return device_info

    def _load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.opt,
            self.lr_scheduler,
            self.early_stopping,
            self.device_,
        )

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1

        # Restore best-model tracking on the checkpoint callback.
        checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)
        if checkpoint_cb is not None and self.early_stopping is not None:
            checkpoint_cb.best_metric = self.early_stopping.best_score
            checkpoint_cb.best_epoch = self.early_stopping.best_epoch
            best_path = os.path.join(self.log_dir, checkpoint_cb.best_filename or "")
            if os.path.isfile(best_path):
                checkpoint_cb.best_model_state = (
                    CheckpointManager.read_model_state_dict(best_path)
                )

        self._resumed = True
        logger.info(f"Resumed training from epoch {self.start_epoch}")

    def _init_metric_logger(self):
        """Initialize the metric logging backend.

        Local CSV logging is always enabled; cloud tracking (SwanLab or
        W&B) is included unless ``--general.cloud_tracking false`` was set.
        """
        from utils.config import config_to_dict

        experiment_name = os.path.basename(self.log_dir) if self.log_dir else "run"
        config = config_to_dict(self.run_config) if self.run_config is not None else {}
        self.metric_logger.init_run(
            log_dir=self.log_dir,
            experiment_name=experiment_name,
            group=self.model.__class__.__name__,
            tags=["cuda" if torch.cuda.is_available() else "cpu"],
            config=config,
        )

    def _finish_metric_logger(self):
        """Finalize and shut down the metric logging backend."""
        self.metric_logger.finish()
        logger.debug("Metric logging finished")

    def run(self):
        """Run the full training loop.

        Initializes metric logging, runs the training loop, and
        finalizes logging and checkpoints.
        """
        if not self._built:
            raise RuntimeError(
                "Trainer has not been built. Please call build() explicitly "
                "before run()."
            )

        self._run_start_time = time.perf_counter()
        self._init_metric_logger()

        try:
            self._train_core()
        finally:
            # Finalize even when training raises: an un-finished run stays
            # active, and SwanLab's login refuses to run while a run is active,
            # which would disable tracking for every subsequent optuna trial.
            self._finish()

    def _train_core(self) -> None:
        """Training core. Single stage runs one epoch loop; multi-stage trainers override this to chain multiple stages."""
        self._run_training_loop()

    def _run_training_loop(self, start_epoch: int | None = None) -> StageResult:
        """Run the epoch training loop for a single stage.

        Includes the epoch loop, callbacks, learning rate scheduling, and
        early stopping; live rendering is owned by ``ProgressCallback``.
        Multi-stage trainers call this after switching ``self.model`` /
        ``self.opt`` / ``self.train_data`` via ``_apply_stage``.

        Args:
            start_epoch: Starting epoch (``None`` uses ``self.start_epoch``,
                used for checkpoint resume).

        Returns:
            A :class:`StageResult` for this stage.
        """
        if start_epoch is None:
            start_epoch = self.start_epoch

        self.model.to(self.device_)
        self.loss = self.loss.to(self.device_)

        # Trigger training start callback
        self.callback_manager.on_train_begin(self.epochs, trainer=self)

        # Stage prefix for multi-stage logging
        stage_prefix = (
            f"[{self._current_stage.upper()}] " if self._current_stage else ""
        )

        # Get monitor name from checkpoint callback
        checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)
        monitor_name = (
            checkpoint_cb._monitor_name() if checkpoint_cb else self._monitor_name()
        )

        epoch = start_epoch
        for epoch in range(start_epoch, self.epochs):
            logger.info(f"{stage_prefix}Epoch {epoch + 1}/{self.epochs}")
            epoch_start = time.perf_counter()

            self.callback_manager.on_epoch_begin(epoch, trainer=self)

            # Training phase
            phase_start = time.perf_counter()
            train_loss = self._process_epoch(epoch, is_train=True)
            train_time = time.perf_counter() - phase_start

            # Validation phase
            val_loss = None
            val_time = 0.0
            if self.val_data is not None:
                phase_start = time.perf_counter()
                val_loss = self._process_epoch(epoch, is_train=False)
                val_time = time.perf_counter() - phase_start

            self.callback_manager.on_epoch_end(
                epoch, train_loss, val_loss, trainer=self
            )

            # Record epoch time (train/val/total)
            epoch_time = time.perf_counter() - epoch_start
            self._epoch_times.append(epoch_time)
            self.metric_logger.log_timing(
                step=epoch + self._metric_step_offset,
                epoch=epoch,
                timings={
                    "train_time": train_time,
                    "val_time": val_time,
                    "epoch_time": epoch_time,
                },
                stage=self._current_stage,
            )
            logger.info(
                f"{stage_prefix}Epoch {epoch + 1}/{self.epochs} took "
                f"{epoch_time:.2f}s (train {train_time:.2f}s, val {val_time:.2f}s)"
            )

            # Learning rate scheduler step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Check early stopping
            if self.callback_manager.should_stop(trainer=self):
                logger.info(
                    f"{stage_prefix}Early stopping triggered at epoch {epoch + 1}"
                )
                break

        logger.info("Training complete")
        self._train_end_time = time.perf_counter()
        self.callback_manager.on_train_end(trainer=self)

        return StageResult(
            name=self._current_stage,
            best_metric=checkpoint_cb.best_metric if checkpoint_cb else None,
            best_epoch=checkpoint_cb.best_epoch if checkpoint_cb else None,
            final_epoch=epoch,
            monitor=monitor_name,
        )

    def _process_epoch(self, epoch: int, is_train: bool) -> float:
        """Process a single epoch of training or validation.

        Args:
            epoch: Current epoch number.
            is_train: Whether this is a training (vs validation) epoch.

        Returns:
            Sample-weighted mean loss for this epoch.
        """
        phase = "train" if is_train else "val"
        data_loader = self.train_data if is_train else self.val_data

        self.metrics_accumulator.reset(phase)
        self.model.train() if is_train else self.model.eval()
        self.callback_manager.on_phase_begin(epoch, phase, trainer=self)

        weighted_loss_sum = 0.0
        total_samples = 0
        for batch_idx, batch_data in enumerate(data_loader):
            self.callback_manager.on_batch_begin(epoch, batch_idx, phase, trainer=self)

            with torch.set_grad_enabled(is_train):
                if is_train:
                    loss, n_samples = self._run_train_batch(batch_data)
                else:
                    loss, n_samples = self._run_eval_batch(batch_data)

            # Weight by valid element count so a short final partial batch
            # cannot dominate the epoch average (and skew cross-batch-size
            # Optuna loss comparison).
            weighted_loss_sum += loss * n_samples
            total_samples += n_samples

            # Log per-batch loss (optional) and update global step
            if self.run_config.general.log_batch_metrics:
                self.metric_logger.log_batch(
                    phase=phase,
                    global_step=self._global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=loss,
                    stage=self._current_stage,
                )
            if is_train:
                self._global_step += 1

            self.callback_manager.on_batch_end(
                epoch, batch_idx, phase, loss, trainer=self
            )

        # Aggregate and log metrics
        metrics = self.metrics_accumulator.compute(phase)
        # Sample-weighted mean is comparable across batch sizes; expose via
        # metrics["loss"] for logging + callbacks.
        mean_loss = weighted_loss_sum / total_samples if total_samples > 0 else 0.0
        metrics["loss"] = mean_loss
        self.metric_logger.log_metrics(
            phase=phase,
            metrics=metrics,
            step=epoch + self._metric_step_offset,
            epoch=epoch,
            stage=self._current_stage,
        )

        self.callback_manager.on_phase_end(
            epoch, phase, mean_loss, metrics, trainer=self
        )

        return mean_loss

    def compute_train_step(
        self, batch_data: tuple[Any, ...]
    ) -> tuple[dict, torch.Tensor]:
        """Execute one training step's pure computation.

        ``zero_grad -> forward_pass -> _compute_loss -> backward -> clip -> step``.
        Shared by the training loop (:meth:`_run_train_batch`) and the efficiency
        benchmark, so the two never drift apart. Excludes the metrics accumulator
        and ``loss.item()`` so callers (the benchmark) can measure throughput
        cleanly. Returns ``(output, loss_tensor)``.
        """
        self.opt.zero_grad(set_to_none=True)
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)
        loss.backward()
        if self.max_clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_clip_grad_norm
            )
        self.opt.step()
        return output, loss

    def _run_train_batch(self, batch_data: tuple[Any, ...]) -> tuple[float, int]:
        """Execute a single training batch.

        Performs forward pass, loss computation, backpropagation, gradient
        clipping, and optimizer step via :meth:`compute_train_step`, then records
        train metrics.

        Args:
            batch_data: A batch of training data.

        Returns:
            ``(loss, n_samples)``: the per-batch mean loss (Python scalar) and
            the number of valid elements it was averaged over, so the epoch
            loop can weight by sample count.
        """
        output, loss = self.compute_train_step(batch_data)
        self.metrics_accumulator.update("train", output)
        return loss.item(), output["y_label"].numel()

    @torch.inference_mode()
    def _run_eval_batch(self, batch_data: tuple[Any, ...]) -> tuple[float, int]:
        """Execute a single evaluation (validation) batch.

        Args:
            batch_data: A batch of validation data.

        Returns:
            ``(loss, n_samples)``: the per-batch mean loss (Python scalar) and
            the number of valid elements it was averaged over.
        """
        output = self.forward_pass(batch_data)
        loss = self._compute_eval_loss(output)

        self.metrics_accumulator.update("val", output)

        return loss.item(), output["y_label"].numel()

    @torch.inference_mode()
    def _run_test_batch(self, batch_data: tuple[Any, ...]) -> float:
        """Execute a single test batch.

        Uses :meth:`test_forward_pass` instead of :meth:`forward_pass` to
        support test-specific logic.

        Args:
            batch_data: A batch of test data from the DataLoader.

        Returns:
            Loss value for this batch (Python scalar).
        """
        output = self.test_forward_pass(batch_data)
        loss = self._compute_eval_loss(output)

        self.metrics_accumulator.update("test", output)

        return loss.item()

    def _iter_eval_batches(self, loader: Any, description: str):
        """Iterate ``loader`` under a progress bar when rendering is enabled.

        Args:
            loader: Test/eval data loader (any sized iterable).
            description: Progress bar label.

        Yields:
            Each batch from ``loader``.
        """
        if not self._progress_enabled:
            yield from loader
            return
        progress = create_progress()
        with progress:
            task = progress.add_task(description, total=len(loader))
            for batch_data in loader:
                yield batch_data
                progress.advance(task)

    @torch.inference_mode()
    def _evaluate_on_test_set(self, use_best_model: bool = True) -> dict[str, float]:
        """Evaluate the model on the test set after training.

        Optionally loads the best model before evaluation.

        Args:
            use_best_model: Whether to load the best checkpoint first.

        Returns:
            Dictionary of test metrics (e.g. auc, acc, rmse).
            Empty dict if test data is not available.
        """
        if self.test_data is None:
            logger.info("Test data not provided. Skipping test evaluation.")
            return {}

        # Check if test set is empty
        test_len = len(self.test_data) if hasattr(self.test_data, "__len__") else None
        if test_len is not None and test_len == 0:
            logger.warning(
                "Test DataLoader is empty (0 batches). Skipping test evaluation. "
                "Cause: The test dataset contains no samples. "
                "During data preprocessing, add_kfold_labels(test_ratio=...) "
                "assigned 0 users to the test fold (fold=-1). "
                "Training and validation completed successfully, but no test "
                "metrics will be recorded."
            )
            return {}

        best_state = None
        if use_best_model and self.early_stopping is not None:
            checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)
            if checkpoint_cb is not None and checkpoint_cb.best_model_state is not None:
                best_state = checkpoint_cb.best_model_state

        if best_state is not None:
            current_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            self.model.load_state_dict(best_state)
        else:
            current_state = None

        self.metrics_accumulator.reset("test")
        self.model.eval()

        for batch_data in self._iter_eval_batches(
            self.test_data, "[bold magenta]Testing"
        ):
            self._run_test_batch(batch_data)

        metrics = self.metrics_accumulator.compute("test")
        self.metric_logger.log_metrics(
            phase="test", metrics=metrics, step=self.epochs or 0, epoch=self.epochs or 0
        )

        if metrics:
            metrics_str = ", ".join(
                f"{name.upper()}={value:.4f}" for name, value in metrics.items()
            )
            logger.info(f"Test metrics: {metrics_str}")

        if current_state is not None:
            self.model.load_state_dict(current_state)

        return metrics

    def load_weights(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file.

        Handles both plain state_dict files (``best_model.pth``) and full
        checkpoint files (``last_checkpoint.pth``).  Delegates to
        :meth:`CheckpointManager.load_weights`.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        self.checkpoint_manager.load_weights(checkpoint_path, self.model, self.device_)
        self.model.to(self.device_)

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the test set and return metrics.

        Call ``load_weights()`` first to load trained weights.

        Returns:
            Dictionary with metric names and values (auc, acc, rmse).
            Empty dict if test data is not available.
        """
        if not self._built:
            raise RuntimeError("Trainer has not been built. Call build() first.")
        if self.test_data is None:
            logger.warning(
                "Test data not available. Ensure build_components returns test_data."
            )
            return {}

        self.model.eval()
        self.metrics_accumulator.reset("test")

        for batch_data in self._iter_eval_batches(
            self.test_data, "[bold magenta]Evaluating"
        ):
            self._run_test_batch(batch_data)

        metrics = self.metrics_accumulator.compute("test")
        self.metric_logger.log_metrics(
            phase="test", metrics=metrics, step=self.epochs or 0, epoch=self.epochs or 0
        )

        if metrics:
            metrics_str = ", ".join(
                f"{name.upper()}={value:.4f}" for name, value in metrics.items()
            )
            logger.info(f"Evaluation results: {metrics_str}")
        else:
            logger.warning("No metrics computed during evaluation.")

        return metrics

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """Compute the training loss from model outputs.

        Subclass overrides may add auxiliary terms (regularization,
        contrastive/multi-task losses); those apply to training only —
        evaluation logging goes through :meth:`_compute_eval_loss`.

        Args:
            outputs: Dict containing ``"y_hat"`` and ``"y_label"``.

        Returns:
            Loss tensor.
        """
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)

    def _compute_eval_loss(self, outputs: dict) -> torch.Tensor:
        """Compute the loss used for val/test logging.

        Pure prediction loss by default: auxiliary terms that a subclass adds
        in ``_compute_loss`` must not leak into val/test loss (they are
        training-only regularizers and would make eval loss incomparable to
        the logged prediction quality). Override only when eval loss should
        deliberately include extra terms.

        Reduced to a per-batch mean: loss fns built with
        ``reduction="none"`` (which reduce inside ``_compute_loss``) still
        yield a scalar here; mean-reduced loss fns are unaffected.

        Args:
            outputs: Dict containing ``"y_hat"`` and ``"y_label"``.

        Returns:
            Loss tensor (scalar).
        """
        return self.loss(outputs["y_hat"], outputs["y_label"]).mean()

    def _monitor_name(self) -> str:
        """Get the name of the metric being monitored for early stopping.

        Returns:
            Metric name string, defaulting to ``"auc"``.
        """
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _print_timing_summary(self) -> None:
        """Print total training time and average time per epoch."""
        # When training raised before completing, _train_end_time is unset;
        # bail out instead of throwing a TypeError from _finish that masks
        # the original exception.
        if self._run_start_time is None or self._train_end_time is None:
            return
        total = self._train_end_time - self._run_start_time
        n_epochs = len(self._epoch_times)
        avg = total / n_epochs if n_epochs else 0.0
        logger.info(
            f"Total time: {datetime.timedelta(seconds=int(total))} | {n_epochs} epochs | "
            f"avg {avg:.2f}s/epoch"
        )

    def _finish(self):
        """Clean up resources and finalize experiment tracking."""
        self._print_timing_summary()
        # Runs on failed runs too (run()'s finally): stops live rendering
        # that on_train_end would have stopped on the normal path.
        if self.callback_manager is not None:
            self.callback_manager.close()
        self._finish_metric_logger()
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.close()

    def release(self) -> None:
        """Shut down DataLoader workers and drop the loader references.

        Persistent workers keep their pipe file descriptors open until the
        owning DataLoader is garbage-collected; a run that exits through an
        exception (e.g. a pruned trial) can delay that collection
        indefinitely, so multi-run processes call this between runs to
        release the workers deterministically.
        """
        for name in ("train_data", "val_data", "test_data"):
            loader = getattr(self, name, None)
            if not isinstance(loader, torch.utils.data.DataLoader):
                continue
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
            setattr(self, name, None)


__all__ = ["BaseTrainer", "StageResult"]
