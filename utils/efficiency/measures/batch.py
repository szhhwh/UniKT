"""Batch-level helpers: shape probing, valid-interaction counting, device moves."""

from collections.abc import Callable
from typing import Any

import torch

from ..device import synchronize
from ..target import BenchmarkTarget


def batch_size_of(batch: Any) -> int:
    """Number of rows (student sequences B) in a batch."""
    first = _first_tensor(batch)
    return int(first.size(0)) if first is not None and first.dim() >= 1 else 0


def _first_tensor(batch: Any) -> torch.Tensor | None:
    """First tensor found in a dict/tuple/list batch (depth 1)."""
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v
    elif isinstance(batch, (tuple, list)):
        for v in batch:
            if isinstance(v, torch.Tensor):
                return v
    return None


def _count_scored(
    forward: Callable[[], dict[str, torch.Tensor]], device: torch.device
) -> int:
    """``numel`` of the forward's aligned 1D ``y_label``, synchronized."""
    # no_grad (not inference_mode): counting forwards may be a model's first
    # execution, and models with lazily built seq-len constants (AKT family)
    # cache them here. Tensors created under inference_mode are rejected by
    # autograd in the later grad-enabled FLOPs/train stages; no_grad tensors
    # are not.
    with torch.no_grad():
        out = forward()
        n = int(out["y_label"].numel())
    synchronize(device)
    return n


def count_valid_interactions(target: BenchmarkTarget, sample_batch: Any) -> int:
    """Valid interactions per forward pass that participate in the loss.

    Throughput denominator: runs one forward via the target, takes ``numel`` of
    the aligned+masked 1D ``y_label`` — the interactions retained by
    ``_extract_valid_predictions`` after the adjacent-pair mask, i.e. the samples
    ``_compute_loss`` actually consumes.
    """
    return _count_scored(lambda: target.forward(sample_batch), target.device)


def count_valid_interactions_split(
    target: BenchmarkTarget, loader: Any
) -> tuple[int, int]:
    """Total valid interactions across one full pass of ``loader``.

    Sums :func:`count_valid_interactions` over every batch and returns
    ``(total, batch_count)``. A full-pass sum is invariant to the loader's
    shuffle order, so dividing ``total`` by ``batch_count`` yields a per-batch
    average that depends only on the dataset and the model's data granularity
    (question- vs skill-level), not on which batch a shuffled loader happens to
    yield first.
    """
    total = 0
    batches = 0
    for batch in loader:
        total += count_valid_interactions(target, to_device(batch, target.device))
        batches += 1
    return total, batches


def count_test_predictions(target: BenchmarkTarget, sample_batch: Any) -> int:
    """Scored predictions per test forward pass.

    Same contract as :func:`count_valid_interactions` but through
    ``test_forward_pass``. For windowlate data this is far smaller than the
    training count — each window scores only its final position — so the two
    must never share a denominator.
    """
    return _count_scored(lambda: target.test_forward(sample_batch), target.device)


def to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move batch tensors to device (tuple/list/dict aware)."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch
