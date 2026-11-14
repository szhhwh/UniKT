"""Windowlate stage: efficiency of the sliding-window evaluation path.

Skill-level models are scored with windowlate data, where every sample carries a
full history but is evaluated at its final position only. Their real serving cost
is therefore one forward pass per *single* prediction, while the ``inference``
stage measures validation sequences that yield a prediction per timestep.
Reporting only the latter overstates skill-level throughput by roughly the
sequence length, so this stage measures the test path on its own terms and
reports the amortization gap explicitly.

The stage owns fetching its own test batch: nothing is loaded unless this stage
actually runs, and the batch is released when it finishes, so other stages never
pay for (or measure around) the test path. Skipped (with the real reason on the
report) when the target has no test loader, its test dataset is not windowlate —
question-level models score dense sequences and are already covered by
``inference``. Data and forward failures propagate to the session's stage guard.
"""

from dataclasses import dataclass
from typing import Any

import torch
from rich.table import Table
from torch.utils.data import DataLoader

from utils.core import get_logger, register_efficiency_stage
from utils.model_data.windowlate import WindowlateIterableDataset

from ..measures.batch import batch_size_of, count_test_predictions, to_device
from ..measures.timing import (
    LatencyMetricsBase,
    benchmark_forward_loop,
)
from ..target import BenchmarkTarget
from .base import EfficiencyStage, StageContext

logger = get_logger(__name__)


@dataclass
class WindowlateMetrics(LatencyMetricsBase):
    """Windowlate evaluation-path efficiency metrics."""

    supported: bool = True
    skip_reason: str = ""
    iters: int = 0
    repeats: int = 0
    data_split: str = "test"
    count_basis: str = "measured_batch"
    inference_batch_size: int | None = None
    test_batch_size: int = 0
    predictions_per_batch: int = 0
    inference_tokens_per_batch: int | None = None
    amortization_ratio: float | None = None
    throughput_predictions_per_sec: float = 0.0
    us_per_prediction: float = 0.0


@dataclass
class WindowlateStageConfig:
    """Windowlate stage knobs."""

    iters: int = 200
    repeats: int = 3


def benchmark_windowlate(
    target: BenchmarkTarget,
    test_batch: Any,
    warmup_iters: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> WindowlateMetrics:
    """Latency/throughput of the windowlate evaluation forward pass.

    Same timing rig as :func:`benchmark_inference` (both go through
    ``benchmark_forward_loop``); the stages differ only in which path they
    exercise and what the throughput denominator counts.
    """
    target.model.eval()
    test_batch_size = batch_size_of(test_batch)
    predictions = count_test_predictions(target, test_batch)
    if predictions <= 0:
        raise ValueError("Test forward produced no scored predictions")
    stats = benchmark_forward_loop(
        lambda: target.test_forward(test_batch),
        warmup_iters,
        iters,
        repeats,
        device,
    )
    wall = stats.sustained_wall_s
    throughput = (predictions * iters) / wall if wall > 0 else 0.0
    us_per = (
        (wall * 1e6) / (predictions * iters) if predictions > 0 and iters > 0 else 0.0
    )
    logger.info(
        f"[Windowlate] latency_mean={stats.latency_mean_ms:.3f}ms "
        f"latency_p95={stats.latency_p95_ms:.3f}ms "
        f"repeat_cv={stats.latency_repeat_cv:.3f} | "
        f"throughput={throughput:,.0f} pred/s "
        f"| {predictions} pred/batch"
        + (
            f" | gpu_peak={stats.gpu_peak_allocated_mib:.0f} MiB"
            if stats.gpu_peak_allocated_mib is not None
            else ""
        )
    )

    return WindowlateMetrics(
        iters=iters,
        repeats=repeats,
        test_batch_size=test_batch_size,
        predictions_per_batch=predictions,
        throughput_predictions_per_sec=throughput,
        us_per_prediction=us_per,
        **LatencyMetricsBase.stats_kwargs(stats),
    )


@register_efficiency_stage("windowlate")
class WindowlateStage(EfficiencyStage):
    """Windowlate evaluation-path efficiency: per-prediction latency and throughput."""

    name = "windowlate"
    priority = 25
    config_cls = WindowlateStageConfig

    def run(self, ctx: StageContext) -> WindowlateMetrics:
        """Benchmark the windowlate test path, or record why it was skipped."""
        loader = ctx.target.test_data
        if loader is None:
            return self._skip("target has no test loader")
        dataset = getattr(loader, "dataset", None)
        if not isinstance(dataset, WindowlateIterableDataset):
            # The dataset type is the only reliable marker: question-level test
            # batches can carry just as many fields as windowlate ones.
            return self._skip(
                f"test dataset is {type(dataset).__name__}, not windowlate "
                "(question-level models are covered by the inference stage)"
            )
        # A single-batch probe loader: iterating the real test loader would
        # fork its persistent workers (each scanning the parquet) and leave
        # them resident, polluting later stages' resource sampling.
        probe = DataLoader(
            dataset,
            batch_size=loader.batch_size,
            num_workers=0,
            collate_fn=loader.collate_fn,
        )
        batch = to_device(next(iter(probe)), ctx.device)
        cfg = ctx.stage_cfg(self.name)
        result = benchmark_windowlate(
            ctx.target,
            batch,
            ctx.general.warmup_iters,
            cfg.iters,
            cfg.repeats,
            ctx.device,
        )
        inference = ctx.results.get("inference")
        if inference is not None:
            result.inference_batch_size = inference.batch_size
            result.inference_tokens_per_batch = inference.valid_tokens_per_batch
            result.amortization_ratio = (
                inference.valid_tokens_per_batch / inference.batch_size
            ) / (result.predictions_per_batch / result.test_batch_size)
        return result

    @staticmethod
    def _skip(reason: str) -> WindowlateMetrics:
        logger.info(f"[Windowlate] skipped: {reason}")
        return WindowlateMetrics(supported=False, skip_reason=reason)

    @classmethod
    def format_table(cls, result: WindowlateMetrics) -> Table | None:
        """Render windowlate per-prediction latency/throughput as a Rich table."""
        table = cls.make_kv_table("Windowlate (evaluation path)")
        if not result.supported:
            table.add_row("Skipped", result.skip_reason)
            return table
        table.add_row("Iterations", f"{result.iters} x {result.repeats}")
        table.add_row("Data split", result.data_split)
        table.add_row("Batch size", f"{result.test_batch_size:,}")
        table.add_row("Count basis", result.count_basis)
        table.add_row("Predictions / batch", f"{result.predictions_per_batch:,}")
        if result.amortization_ratio is not None:
            table.add_row(
                "Amortization vs inference path",
                f"x{result.amortization_ratio:,.1f} per sample",
            )
        cls.add_latency_rows(table, result)
        table.add_row(
            "Throughput (sustained)",
            f"{result.throughput_predictions_per_sec:,.0f} predictions/s",
        )
        table.add_row("Per prediction", f"{result.us_per_prediction:,.1f} us")
        return table
