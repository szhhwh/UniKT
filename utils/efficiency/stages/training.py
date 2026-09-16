"""Training stage: peak memory + throughput via a pseudo train loop."""

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import torch
from rich.table import Table

from utils.core import get_logger, register_efficiency_stage

from ..device import DeviceBackend
from ..measures.train_step import run_train_step
from ..target import BenchmarkTarget
from .base import EfficiencyStage, StageContext, format_duration, format_valid_tokens

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Training efficiency metrics (measured via a pseudo train loop)."""

    iters: int = 0
    repeats: int = 0
    batch_size: int = 0
    valid_tokens_per_batch: float = 0.0
    valid_tokens_total: int = 0
    valid_tokens_batches: int = 0
    wall_time_s: float = 0.0
    ms_per_train_step: float = 0.0
    repeat_ms_per_step: list[float] = field(default_factory=list)
    throughput_interactions_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    ns_per_interaction: float = 0.0
    gpu_peak_allocated_mib: float | None = None
    gpu_peak_reserved_mib: float | None = None


@dataclass
class TrainStageConfig:
    """Training stage knobs."""

    iters: int = 50
    repeats: int = 1


def benchmark_training(
    target: BenchmarkTarget,
    sample_batch: Any,
    batch_size: int,
    valid_tokens: float,
    valid_tokens_total: int,
    valid_tokens_batches: int,
    warmup_iters: int,
    iters: int,
    device: torch.device,
    repeats: int = 1,
) -> TrainingMetrics:
    """Training peak memory + throughput benchmark.

    Runs ``zero_grad -> forward_pass -> _compute_loss -> backward -> clip -> step``
    via :func:`run_train_step`, which delegates to ``target.compute_train_step``
    — the same computation the real training loop performs. The metrics
    accumulator is bypassed to keep throughput measurement clean. ``valid_tokens``
    is the per-batch average over a full train-split pass (the timing loop itself
    reuses ``sample_batch``; the uniform-input override pins every batch to the
    same padded shape, so its step cost represents any batch).

    Warmup runs once; the timed loop is repeated ``repeats`` times and the
    reported step time is the median across repeats — launch-bound steps have
    heavy host-side jitter on shared CPUs, which a single short mean cannot
    resolve.
    """
    model = target.model
    model.train()
    dev = DeviceBackend(device)

    # warmup: includes backward pass to fill cudnn backward autotune + Adam momentum to steady state
    for _ in range(warmup_iters):
        run_train_step(target, sample_batch)
    dev.sync()

    repeat_ms: list[float] = []
    total_wall = 0.0
    with dev.peak_memory() as mem:
        for _ in range(max(repeats, 1)):
            start = time.perf_counter()
            for _ in range(iters):
                run_train_step(target, sample_batch)
            dev.sync()
            wall = time.perf_counter() - start
            total_wall += wall
            repeat_ms.append(wall / iters * 1000 if iters > 0 else 0.0)

    peak_alloc = mem.allocated_mib
    peak_reserved = mem.reserved_mib

    ms_per_step = median(repeat_ms) if repeat_ms else 0.0
    # ms_per_step is per single step, so the iters factor cancels out here
    throughput = valid_tokens / (ms_per_step / 1000) if ms_per_step > 0 else 0.0
    samples_per_sec = batch_size / (ms_per_step / 1000) if ms_per_step > 0 else 0.0
    ns_per = (
        ms_per_step * 1e6 / valid_tokens
        if valid_tokens > 0 and ms_per_step > 0
        else 0.0
    )

    logger.info(
        f"[Training] step={ms_per_step:.3f}ms (repeats: "
        + ", ".join(f"{v:.3f}" for v in repeat_ms)
        + f"ms) | throughput={throughput:,.0f} int/s"
        + (f" | gpu_peak={peak_alloc:.0f} MiB" if peak_alloc is not None else "")
    )

    return TrainingMetrics(
        iters=iters,
        repeats=max(repeats, 1),
        batch_size=batch_size,
        valid_tokens_per_batch=valid_tokens,
        valid_tokens_total=valid_tokens_total,
        valid_tokens_batches=valid_tokens_batches,
        wall_time_s=total_wall,
        ms_per_train_step=ms_per_step,
        repeat_ms_per_step=repeat_ms,
        throughput_interactions_per_sec=throughput,
        samples_per_sec=samples_per_sec,
        ns_per_interaction=ns_per,
        gpu_peak_allocated_mib=peak_alloc,
        gpu_peak_reserved_mib=peak_reserved,
    )


@register_efficiency_stage("train")
class TrainingStage(EfficiencyStage):
    """Training efficiency: peak memory + throughput via a pseudo train loop."""

    name = "train"
    priority = 30
    config_cls = TrainStageConfig

    def run(self, ctx: StageContext) -> TrainingMetrics:
        """Benchmark training peak memory and throughput over a pseudo loop."""
        cfg = ctx.stage_cfg(self.name)
        return benchmark_training(
            ctx.target,
            ctx.sample_batch,
            ctx.batch_size,
            ctx.valid_tokens,
            ctx.valid_tokens_total,
            ctx.valid_tokens_batches,
            ctx.general.warmup_iters,
            cfg.iters,
            ctx.device,
            cfg.repeats,
        )

    @classmethod
    def format_table(cls, result: TrainingMetrics) -> Table | None:
        """Render training throughput/memory as a Rich table."""
        table = cls.make_kv_table("Training (pseudo loop)")
        table.add_row("Iterations", f"{result.iters} x {result.repeats}")
        table.add_row("Per step", f"{result.ms_per_train_step:.3f} ms")
        table.add_row(
            "Valid tokens / batch",
            format_valid_tokens(
                result.valid_tokens_per_batch,
                result.valid_tokens_total,
                result.valid_tokens_batches,
            ),
        )
        table.add_row(
            "Throughput (sustained)",
            f"{result.throughput_interactions_per_sec:,.0f} interactions/s",
        )
        table.add_row("Samples/s", f"{result.samples_per_sec:,.0f}")
        table.add_row("Wall time", format_duration(result.wall_time_s))
        if result.gpu_peak_allocated_mib is not None:
            table.add_row(
                "GPU peak (allocated)", f"{result.gpu_peak_allocated_mib:,.0f} MiB"
            )
        if result.gpu_peak_reserved_mib is not None:
            table.add_row(
                "GPU peak (reserved)", f"{result.gpu_peak_reserved_mib:,.0f} MiB"
            )
        return table
