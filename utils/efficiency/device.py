"""Device abstraction: the single home for CUDA-vs-CPU branching in the benchmark.

Centralizes synchronization, peak-memory windows, and per-step event timing so
stages never sprinkle ``if device.type == "cuda"`` ladders.
"""

import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MemoryPeak:
    """Peak GPU memory observed during a measurement window (MiB)."""

    allocated_mib: float | None = None
    reserved_mib: float | None = None


def synchronize(device: torch.device) -> None:
    """Wait for all queued CUDA kernels; no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reclaim_memory(device: torch.device) -> None:
    """Best-effort reclaim after a failed measurement (e.g. CUDA OOM recovery).

    The allocator keeps failed-stage blocks reserved; without this the next
    stage inherits the fragmentation and fails identically.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class DeviceBackend:
    """Wraps a ``torch.device`` with the measurement primitives stages need."""

    def __init__(self, device: torch.device) -> None:
        """Bind the backend to a device and cache whether it is CUDA."""
        self.device = device
        self.is_cuda = device.type == "cuda"

    def sync(self) -> None:
        """Wait for all queued kernels; no-op on CPU."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def peak_memory(self) -> Iterator[MemoryPeak]:
        """Reset peak-memory stats on enter, read them on exit (MiB).

        Wraps the section whose peak allocation is measured. No-op (yields an
        empty :class:`MemoryPeak`) on CPU. Reuses the reset/read pair previously
        duplicated across the inference, training, and trace stages.
        """
        peak = MemoryPeak()
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        yield peak
        if self.is_cuda:
            peak.allocated_mib = torch.cuda.max_memory_allocated(self.device) / 1024**2
            peak.reserved_mib = torch.cuda.max_memory_reserved(self.device) / 1024**2

    def time_step_events(self, step: Callable[[], Any]) -> float:
        """Time one ``step`` call: CUDA events on GPU (ms), ``perf_counter`` on CPU."""
        if self.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)
        t0 = time.perf_counter()
        step()
        return (time.perf_counter() - t0) * 1000
