"""Checkpoint management module.

Provides model checkpoint save and load functionality.
"""

import atexit
import os
import random
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import Any, cast

import numpy as np
import torch

from ..core import get_logger
from .early_stopping import EarlyStopping

logger = get_logger(__name__)


def _detach_to_cpu(obj: Any) -> Any:
    """Recursively clone tensors in a state dict to CPU.

    Works for model, optimizer, and scheduler state dicts.

    Args:
        obj: A tensor, dict, list, or tuple potentially containing tensors.

    Returns:
        The same structure with all tensors detached, moved to CPU,
        and cloned.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def _strip_compile_prefix_if_needed(
    state_dict: dict[str, Any], model: torch.nn.Module
) -> dict[str, Any]:
    # Strip the `_orig_mod.` prefix torch.compile adds when a checkpoint
    # saved from a compiled model is loaded into a raw model.
    model_prefixed = any(k.startswith("_orig_mod.") for k in model.state_dict())
    state_prefixed = any(k.startswith("_orig_mod.") for k in state_dict)
    if state_prefixed and not model_prefixed:
        return {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return state_dict


def _capture_rng_states() -> dict[str, Any]:
    # numpy's get_state() holds a uint32 ndarray, which is not allowed under
    # torch.load's default weights_only=True. Convert it to a list so the
    # whole checkpoint stays weights-only-safe.
    # numpy stubs model the state as a TypedDict; runtime is a tuple.
    np_state: Any = np.random.get_state()
    np_state_safe = (
        np_state[0],
        np_state[1].tolist(),
        np_state[2],
        np_state[3],
        np_state[4],
    )
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np_state_safe,
        "python": random.getstate(),
    }


def _restore_rng_states(states: dict[str, Any]) -> None:
    # RNG state tensors must live on CPU: torch.load(map_location=device) in
    # load_checkpoint moves every tensor to the target device, but the
    # set_rng_state APIs reject CUDA ByteTensors.
    if states.get("torch") is not None:
        torch.set_rng_state(states["torch"].cpu())
    if torch.cuda.is_available() and states.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])
    if states.get("numpy") is not None:
        np_state = states["numpy"]
        np_state = (
            np_state[0],
            np.asarray(np_state[1], dtype=np.uint32),
            np_state[2],
            np_state[3],
            np_state[4],
        )
        np.random.set_state(np_state)
    if states.get("python") is not None:
        random.setstate(states["python"])


class CheckpointManager:
    """Manager for model checkpoints.

    Responsibilities:
    1. Save model checkpoints.
    2. Load model checkpoints.
    3. Manage checkpoint files.

    Example:
        >>> ckpt_mgr = CheckpointManager(log_dir="./runs/exp1")
        >>> ckpt_mgr.save_checkpoint(
        ...     epoch=10,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     filename="checkpoint.pth"
        ... )
    """

    def __init__(self, log_dir: str):
        """Initialize the checkpoint manager.

        Args:
            log_dir: Directory path for saving checkpoint files.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-saver"
        )
        self._latest: dict[str, Future] = {}
        self._closed = False
        atexit.register(self.close)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        additional_state: dict | None = None,
        early_stopping_state: dict | None = None,
        filename: str = "checkpoint.pth",
    ) -> None:
        """Save a full checkpoint including model, optimizer, and scheduler states.

        Args:
            epoch: Current epoch number.
            model: PyTorch model.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler (optional).
            additional_state: Extra state information (optional).
            early_stopping_state: Early stopping state (optional).
            filename: Checkpoint file name.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": _detach_to_cpu(model.state_dict()),
            "optimizer_state_dict": _detach_to_cpu(optimizer.state_dict()),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = _detach_to_cpu(scheduler.state_dict())

        if early_stopping_state is not None:
            state["early_stopping_state"] = early_stopping_state

        state["rng_states"] = _capture_rng_states()

        if additional_state:
            state.update(additional_state)

        filepath = os.path.join(self.log_dir, filename)
        self._submit_save(state, filepath)

    def save_weights(self, model: torch.nn.Module, filename: str = "model.pth") -> dict:
        """Save model weights only and return a CPU snapshot.

        Args:
            model: PyTorch model.
            filename: Output file name.

        Returns:
            CPU-cloned model state_dict (same content as the file).
        """
        snapshot = cast(dict, _detach_to_cpu(model.state_dict()))
        filepath = os.path.join(self.log_dir, filename)
        self._submit_save(snapshot, filepath)
        return snapshot

    def _submit_save(self, obj: Any, filepath: str) -> None:
        """Submit an atomic save to the background thread.

        Coalesces with the previous save to ``filepath``: if it has not
        started yet it is cancelled (its older data would be overwritten by
        this one anyway); ``Future.cancel`` is a no-op on a running save, so
        an in-flight write is never interrupted. Without coalescing, a slow
        disk lets stale writes pile up in the single-worker queue and drain
        serially — and slowly — at ``close()``, making a finished run look
        stuck. Falls back to a synchronous write if the manager is closed.

        Args:
            obj: Data to save.
            filepath: Destination file path.
        """
        if self._closed:
            self._write_atomic(obj, filepath)
            return
        prev = self._latest.get(filepath)
        if prev is not None and not prev.done():
            prev.cancel()
        future = self._executor.submit(self._write_atomic, obj, filepath)
        future.add_done_callback(self._log_failure)
        self._latest[filepath] = future

    @staticmethod
    def _log_failure(future: Future) -> None:
        """Log a finished save's exception, if any (skip cancelled saves)."""
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.error("Async checkpoint save failed", exc_info=exc)

    @staticmethod
    def _write_atomic(obj: Any, filepath: str) -> None:
        """Atomically write data to a file via temp + replace."""
        tmp = filepath + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def flush(self) -> None:
        """Wait for the latest save per path to finish.

        Exceptions are already surfaced by the done-callback, so they are
        swallowed here. ``close`` follows this with ``shutdown(wait=True)``,
        which also drains any older running save whose future was replaced.
        """
        for future in list(self._latest.values()):
            if future.cancelled():
                continue
            with suppress(Exception):
                future.result()
        self._latest = {p: f for p, f in self._latest.items() if not f.done()}

    def close(self) -> None:
        """Drain the save queue and shut down the executor.

        Idempotent — safe to call from both ``_finish`` and ``atexit``.
        """
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._executor.shutdown(wait=True)

    @staticmethod
    def load_weights(
        path: str,
        model: torch.nn.Module,
        device: torch.device | None = None,
    ) -> dict | None:
        """Load model weights from a checkpoint file.

        Handles both formats:
        - Plain state_dict (saved by ``save_weights``)
        - Full checkpoint dict with ``"model_state_dict"`` key (saved by ``save_checkpoint``)

        Args:
            path: Path to the checkpoint file.
            model: PyTorch model to load weights into.
            device: Target device for ``map_location`` (optional).

        Returns:
            The full checkpoint dict if the file was a full checkpoint,
            ``None`` if it was a plain state_dict.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_kw: dict[str, Any] = {"map_location": device} if device is not None else {}
        logger.info(f"Loading model weights from {path}...")
        raw = torch.load(path, **map_kw)

        if isinstance(raw, dict) and "model_state_dict" in raw:
            model.load_state_dict(
                _strip_compile_prefix_if_needed(raw["model_state_dict"], model)
            )
            logger.info(
                f"Loaded from full checkpoint (epoch {raw.get('epoch', 'unknown')})"
            )
            return raw

        model.load_state_dict(_strip_compile_prefix_if_needed(raw, model))
        logger.debug("Loaded from plain state_dict")
        return None

    @staticmethod
    def read_model_state_dict(path: str, device: torch.device | None = None) -> dict:
        """Read a checkpoint file and return the model state_dict.

        Does not load into a model. Handles both plain state_dict files and
        full checkpoint dicts with a ``"model_state_dict"`` key.
        """
        map_kw: dict[str, Any] = {"map_location": device} if device is not None else {}
        raw = torch.load(path, **map_kw)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            return raw["model_state_dict"]
        return raw

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        early_stopping: EarlyStopping | None = None,
        device: torch.device | None = None,
    ) -> dict:
        """Load a full checkpoint and restore model, optimizer, and scheduler states.

        Args:
            checkpoint_path: Path to the checkpoint file.
            model: PyTorch model.
            optimizer: Optimizer (optional).
            scheduler: Learning rate scheduler (optional).
            early_stopping: Early stopping object (optional).
            device: Compute device for map_location (optional).

        Returns:
            The checkpoint dictionary.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)

        # Restore model state
        model.load_state_dict(
            _strip_compile_prefix_if_needed(checkpoint["model_state_dict"], model)
        )

        # Restore optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Move optimizer state tensors to the target device.
            if device is not None:
                target = torch.device(device)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(target)

        # Restore scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore early stopping state
        if early_stopping is not None and "early_stopping_state" in checkpoint:
            es_state = checkpoint["early_stopping_state"]
            early_stopping.best_score = es_state.get("best_score")
            early_stopping.best_epoch = es_state.get("best_epoch")
            early_stopping.num_bad_epochs = es_state.get("num_bad_epochs", 0)
            early_stopping.best_metrics = es_state.get("best_metrics")

        # Restore RNG states
        if "rng_states" in checkpoint:
            _restore_rng_states(checkpoint["rng_states"])

        logger.info("Checkpoint loaded successfully")
        return checkpoint


__all__ = ["CheckpointManager"]
