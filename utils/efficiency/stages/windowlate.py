"""Windowlate stage: efficiency of the sliding-window evaluation path.

Skill-level models are scored with windowlate data, where every sample carries a
full history but is evaluated at its final position only. Their real serving cost
is therefore one forward pass per *single* prediction, while the ``inference``
stage measures the training-shaped path that yields a prediction per timestep.
Reporting only the latter overstates skill-level throughput by roughly the
sequence length, so this stage measures the test path on its own terms and
reports the amortization gap explicitly.

The stage owns fetching its own test batch: nothing is loaded unless this stage
actually runs, and the batch is released when it finishes, so other stages never
pay for (or measure around) the test path. Skipped (with the real reason on the
report) when the target has no test loader, its test dataset is not windowlate —
question-level models score dense sequences and are already covered by
``inference`` — or the test path fails.
"""

from dataclasses import dataclass
from typing import Any

import torch
from rich.table import Table
from torch.utils.data import DataLoader

from utils.core import get_logger, register_efficiency_stage
from utils.model_data.skill_model_data import WindowlateIterableDataset

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
    train_batch_size: int = 0
    test_batch_size: int = 0
    predictions_per_batch: int = 0
    # Per-batch average over a full train-split pass when known (float); the
    # amortization ratio uses it only as a numerator scale, so the wider type
    # changes no computed value.
    train_path_tokens_per_batch: float = 0.0
    amortization_ratio: float = 0.0
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
    train_batch_size: int,
    test_batch_size: int,
    predictions: int,
    train_tokens: float,
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
    # How much cheaper a prediction looks when the same forward is credited with
    # every timestep instead of the one position windowlate actually scores.
    # Per-sample on both sides: the train and test loaders may batch differently.
    amortization = (
        (train_tokens / train_batch_size) / (predictions / test_batch_size)
        if predictions > 0 and train_batch_size > 0 and test_batch_size > 0
        else 0.0
    )

    logger.info(
        f"[Windowlate] latency_mean={stats.latency_mean_ms:.3f}ms "
        f"latency_p95={stats.latency_p95_ms:.3f}ms "
        f"repeat_cv={stats.latency_repeat_cv:.3f} | "
        f"throughput={throughput:,.0f} pred/s "
        f"| {predictions} pred/batch (per-sample amortization x{amortization:,.1f})"
        + (
            f" | gpu_peak={stats.gpu_peak_allocated_mib:.0f} MiB"
            if stats.gpu_peak_allocated_mib is not None
            else ""
        )
    )

    return WindowlateMetrics(
        iters=iters,
        repeats=repeats,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        predictions_per_batch=predictions,
        train_path_tokens_per_batch=train_tokens,
        amortization_ratio=amortization,
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
            batch_size=getattr(loader, "batch_size", None) or 1,
            num_workers=0,
        )
        try:
            batch = to_device(next(iter(probe)), ctx.device)
        except StopIteration:
            return self._skip("test loader is empty")
        except Exception as exc:
            return self._skip(f"test loader failed: {exc}")

        test_batch_size = batch_size_of(batch)
        if test_batch_size <= 0:
            return self._skip("test batch has no tensor rows")
        try:
            predictions = count_test_predictions(ctx.target, batch)
        except Exception as exc:
            return self._skip(f"test forward failed: {exc}")
        if predictions <= 0:
            return self._skip("test forward produced no scored predictions")

        try:
            cfg = ctx.stage_cfg(self.name)
            return benchmark_windowlate(
                ctx.target,
                batch,
                ctx.batch_size,
                test_batch_size,
                predictions,
                ctx.valid_tokens,
                ctx.general.warmup_iters,
                cfg.iters,
                cfg.repeats,
                ctx.device,
            )
        except Exception as exc:
            # The session has no per-stage guard; a supplemental stage must not
            # take training/trace down with it.
            return self._skip(f"benchmark failed: {exc}")

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
        table.add_row(
            "Train / test batch size",
            f"{result.train_batch_size:,} / {result.test_batch_size:,}",
        )
        table.add_row("Predictions / batch", f"{result.predictions_per_batch:,}")
        table.add_row(
            "Train-path tokens / batch", f"{result.train_path_tokens_per_batch:,.1f}"
        )
        table.add_row(
            "Amortization vs train path",
            f"x{result.amortization_ratio:,.1f} per sample",
        )
        cls.add_latency_rows(table, result)
        table.add_row(
            "Throughput (sustained)",
            f"{result.throughput_predictions_per_sec:,.0f} predictions/s",
        )
        table.add_row("Per prediction", f"{result.us_per_prediction:,.1f} us")
        return table
