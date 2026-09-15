"""Forward FLOPs and disk-size estimation via ``torch.utils.flop_counter``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.utils.flop_counter import FlopCounterMode

from utils.core import get_logger

logger = get_logger(__name__)


def estimate_disk_size_mb(model: torch.nn.Module) -> float:
    """Estimate state_dict serialization size (params + buffers, by element_size)."""
    size_bytes = 0
    for p in model.parameters():
        size_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        size_bytes += b.numel() * b.element_size()
    return size_bytes / 1024**2


def count_flops(
    forward_fn: Callable[[], Any], device: torch.device
) -> tuple[int | None, dict[str, int]]:
    """Count forward FLOPs with ``torch.utils.flop_counter.FlopCounterMode``.

    cuDNN stays enabled: fused ``aten::_cudnn_rnn`` (LSTM/GRU/RNN) is covered by a
    custom formula registered on import of :mod:`utils.efficiency.measures`, so FLOPs
    share the cuDNN config used for latency and memory.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode

        # Grad-enabled (NOT inference_mode/no_grad): FlopCounterMode's ModuleTracker
        # fw_pre_hook registers grad hooks that read grad_fn.next_functions; under
        # inference_mode grad_fn is None -> AttributeError. Forward builds a graph
        # (one pass, acceptable).
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        flops = flop_counter.get_total_flops()
        breakdown = format_breakdown(flop_counter)
        return flops, breakdown
    except Exception as e:
        logger.warning(f"[Profile] FLOPs measurement failed: {e}")
        return None, {}


def format_breakdown(flop_counter: FlopCounterMode) -> dict[str, int]:
    """Top-level aten op to FLOPs mapping from FlopCounterMode, sorted desc."""
    try:
        counts = flop_counter.get_flop_counts()
        global_counts = counts.get("Global", {})
        result: dict[str, int] = {}
        for op, val in global_counts.items():
            key = str(op).split(".")[-1] if "." in str(op) else str(op)
            result[key] = int(val)
        return dict(sorted(result.items(), key=lambda x: -x[1]))
    except Exception:
        return {}
