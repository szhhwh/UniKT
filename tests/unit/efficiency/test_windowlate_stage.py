"""Tests for the windowlate stage: guard-chain skips, per-sample amortization, shared timing rig.

Covers the code-review fixes: dataset-type gating (question-level 6-tuple
batches must skip, LBKT/BDGKT regression), per-sample amortization across
differing train/test batch sizes, no exception escaping ``run`` (loader /
forward / benchmark), and ``benchmark_forward_loop`` being the single timing
rig behind both stages with ``InferenceMetrics``' JSON keys unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from utils.efficiency.environment import EnvironmentInfo
from utils.efficiency.measures.timing import benchmark_forward_loop
from utils.efficiency.stages.base import StageContext
from utils.efficiency.stages.inference import InferenceMetrics, benchmark_inference
from utils.efficiency.stages.windowlate import (
    WindowlateMetrics,
    WindowlateStage,
    WindowlateStageConfig,
)
from utils.model_data.skill_model_data import WindowlateIterableDataset

# Matches the exact 7-tuple sample shape declared by
# WindowlateIterableDataset.__iter__ so the stubs below stay override-safe.
_Sample = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

_LATENCY_KEYS = {
    "latency_mean_ms",
    "latency_std_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_min_ms",
    "latency_max_ms",
    "latency_cv",
    "latency_repeat_std_ms",
    "latency_repeat_cv",
    "per_repeat_mean_ms",
    "gpu_peak_allocated_mib",
    "gpu_peak_reserved_mib",
}

_SAMPLE_TUPLE = (
    torch.zeros(4, dtype=torch.long),
    torch.zeros(4, dtype=torch.long),
    torch.ones(4, dtype=torch.bool),
    torch.zeros(4, dtype=torch.long),
    torch.zeros(4, dtype=torch.long),
    torch.zeros(4, dtype=torch.long),
    torch.zeros(4, dtype=torch.long),
)


class _StubWindowlateDataset(WindowlateIterableDataset):
    """Yield synthetic 7-tuple samples without touching any parquet file."""

    def __init__(self, n_samples: int = 8) -> None:
        super().__init__("unused.parquet", max_seq_len=4)
        self.n_samples = n_samples

    def __iter__(self) -> Iterator[_Sample]:
        for _ in range(self.n_samples):
            yield _SAMPLE_TUPLE


class _EmptyWindowlateDataset(WindowlateIterableDataset):
    def __init__(self) -> None:
        super().__init__("unused.parquet", max_seq_len=4)

    def __iter__(self) -> Iterator[_Sample]:
        yield from ()


class _FailingWindowlateDataset(WindowlateIterableDataset):
    def __init__(self) -> None:
        super().__init__("unused.parquet", max_seq_len=4)

    def __iter__(self) -> Iterator[_Sample]:
        raise FileNotFoundError("windowlate parquet missing")


class _TupleDataset(Dataset):
    """Question-level style map-style dataset: 7-tuple samples, no windowlate."""

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> _Sample:
        return _SAMPLE_TUPLE


class _StubTarget:
    """Duck-typed BenchmarkTarget: configurable test path behaviour + call count."""

    train_data = None

    def __init__(
        self,
        test_data: Any,
        y_label_size: int | None = None,
        error: Exception | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.model = torch.nn.Linear(2, 2)
        self.test_data = test_data
        self.device = torch.device("cpu")
        self._y_label_size = y_label_size
        self._error = error
        self._fail_after = fail_after
        self.calls = 0

    def test_forward(self, batch: Any) -> dict[str, torch.Tensor]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._fail_after is not None and self.calls > self._fail_after:
            raise RuntimeError("boom mid-benchmark")
        n = self._y_label_size if self._y_label_size is not None else batch[0].size(0)
        return {"y_label": torch.zeros(n)}

    forward = test_forward

    def compute_train_step(self, batch: Any) -> tuple[dict[str, Any], torch.Tensor]:
        return {}, torch.zeros(())

    def prepare(self, device: torch.device) -> None:
        return None


def _run_stage(
    target: _StubTarget,
    batch_size: int = 8,
    valid_tokens: int = 80,
    cfg: Any = None,
) -> WindowlateMetrics:
    ctx = StageContext(
        target=target,
        device=torch.device("cpu"),
        sample_batch=None,
        batch_size=batch_size,
        valid_tokens=valid_tokens,
        seq_len=4,
        cfg=cfg
        or SimpleNamespace(
            general=SimpleNamespace(warmup_iters=2),
            windowlate=WindowlateStageConfig(iters=3, repeats=2),
        ),
        environment=EnvironmentInfo(),
    )
    return WindowlateStage().run(ctx)


# ---------------------------------------------------------------------------
# guard chain: every failure mode skips with its real reason, never raises
# ---------------------------------------------------------------------------


def test_skip_no_test_loader() -> None:
    result = _run_stage(_StubTarget(test_data=None))
    assert result.supported is False
    assert result.skip_reason == "target has no test loader"


def test_skip_non_windowlate_dataset_lbt_style_regression() -> None:
    loader = DataLoader(_TupleDataset(), batch_size=4)
    result = _run_stage(_StubTarget(loader))
    assert result.supported is False
    assert "_TupleDataset" in result.skip_reason
    assert "not windowlate" in result.skip_reason


def test_skip_loader_failure_does_not_raise() -> None:
    loader = DataLoader(_FailingWindowlateDataset(), batch_size=4)
    result = _run_stage(_StubTarget(loader))
    assert result.supported is False
    assert result.skip_reason.startswith("test loader failed:")
    assert "windowlate parquet missing" in result.skip_reason


def test_skip_empty_loader() -> None:
    loader = DataLoader(_EmptyWindowlateDataset(), batch_size=4)
    result = _run_stage(_StubTarget(loader))
    assert result.supported is False
    assert result.skip_reason == "test loader is empty"


def test_skip_test_forward_failure_reports_cause() -> None:
    loader = DataLoader(_StubWindowlateDataset(), batch_size=4)
    target = _StubTarget(loader, error=KeyError("y_label"))
    result = _run_stage(target)
    assert result.supported is False
    assert result.skip_reason.startswith("test forward failed:")


def test_skip_zero_predictions() -> None:
    loader = DataLoader(_StubWindowlateDataset(), batch_size=4)
    target = _StubTarget(loader, y_label_size=0)
    result = _run_stage(target)
    assert result.supported is False
    assert result.skip_reason == "test forward produced no scored predictions"


def test_benchmark_failure_does_not_raise() -> None:
    loader = DataLoader(_StubWindowlateDataset(), batch_size=4)
    # Survive count + warmup calls, then die inside the timed loop.
    target = _StubTarget(loader, fail_after=3)
    result = _run_stage(target)
    assert result.supported is False
    assert result.skip_reason.startswith("benchmark failed:")


def test_missing_stage_config_skips_not_crashes() -> None:
    loader = DataLoader(_StubWindowlateDataset(), batch_size=4)
    # Hand-built cfg whose schema predates the windowlate node.
    cfg = SimpleNamespace(general=SimpleNamespace(warmup_iters=2))
    result = _run_stage(_StubTarget(loader), cfg=cfg)
    assert result.supported is False
    assert result.skip_reason.startswith("benchmark failed:")


class _ExplodingLoader:
    """Real-loader stand-in: exposes the dataset but must never be iterated."""

    batch_size = 4

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        raise AssertionError("original test loader iterated")


def test_fetches_via_probe_loader_not_the_real_one() -> None:
    target = _StubTarget(_ExplodingLoader(_StubWindowlateDataset()))
    result = _run_stage(target)
    assert result.supported is True
    assert result.test_batch_size == 4
    assert result.predictions_per_batch == 4


# ---------------------------------------------------------------------------
# happy path: per-sample amortization across differing train/test batch sizes
# ---------------------------------------------------------------------------


def test_windowlate_metrics_and_per_sample_amortization() -> None:
    loader = DataLoader(_StubWindowlateDataset(), batch_size=8)
    # RobustKT-style asymmetry: train batch 64 (640 valid tokens = 10/sample),
    # test batch 8 scoring 1 prediction/sample. The un-normalized ratio would
    # be 640/8 = 80; per-sample it must be 10.
    result = _run_stage(_StubTarget(loader), batch_size=64, valid_tokens=640)
    assert result.supported is True
    assert result.skip_reason == ""
    assert result.test_batch_size == 8
    assert result.train_batch_size == 64
    assert result.predictions_per_batch == 8
    assert result.train_path_tokens_per_batch == 640
    assert result.amortization_ratio == pytest.approx(10.0)
    assert result.iters == 3
    assert result.repeats == 2
    assert result.latency_mean_ms > 0
    assert result.throughput_predictions_per_sec > 0
    assert result.us_per_prediction > 0
    assert result.throughput_predictions_per_sec * result.us_per_prediction == (
        pytest.approx(1e6, rel=1e-6)
    )
    assert result.gpu_peak_allocated_mib is None
    assert set(asdict(result)) == (
        _LATENCY_KEYS
        | {
            "supported",
            "skip_reason",
            "iters",
            "repeats",
            "train_batch_size",
            "test_batch_size",
            "predictions_per_batch",
            "train_path_tokens_per_batch",
            "amortization_ratio",
            "throughput_predictions_per_sec",
            "us_per_prediction",
        }
    )


# ---------------------------------------------------------------------------
# shared timing rig (#7): call accounting + stable InferenceMetrics keys
# ---------------------------------------------------------------------------


def test_benchmark_forward_loop_call_accounting() -> None:
    calls: list[int] = []

    def fwd() -> None:
        calls.append(1)

    stats = benchmark_forward_loop(fwd, 2, 3, 2, torch.device("cpu"))
    # warmup + sustained loop + repeats * per-iteration event timing
    assert len(calls) == 2 + 3 + 2 * 3
    assert stats.sustained_wall_s > 0
    assert stats.latency_mean_ms > 0
    assert len(stats.per_repeat_mean_ms) == 2


def test_inference_metrics_json_keys_unchanged() -> None:
    expected = _LATENCY_KEYS | {
        "iters",
        "repeats",
        "batch_size",
        "valid_tokens_per_batch",
        "valid_tokens_total",
        "valid_tokens_batches",
        "throughput_interactions_per_sec",
        "ns_per_interaction",
    }
    assert set(asdict(InferenceMetrics())) == expected


def test_benchmark_inference_throughput_ns_reciprocal() -> None:
    target = _StubTarget(test_data=None)
    result = benchmark_inference(
        target,
        _SAMPLE_TUPLE,
        batch_size=4,
        valid_tokens=16,
        valid_tokens_total=32,
        valid_tokens_batches=2,
        warmup_iters=2,
        iters=3,
        repeats=2,
        device=torch.device("cpu"),
    )
    assert result.valid_tokens_per_batch == 16
    assert result.throughput_interactions_per_sec > 0
    assert (
        result.throughput_interactions_per_sec * result.ns_per_interaction
        == pytest.approx(1e9, rel=1e-6)
    )


def test_benchmark_inference_enforces_eval_mode() -> None:
    target = _StubTarget(test_data=None)
    target.model.train()
    benchmark_inference(
        target,
        _SAMPLE_TUPLE,
        batch_size=4,
        valid_tokens=16,
        valid_tokens_total=32,
        valid_tokens_batches=2,
        warmup_iters=1,
        iters=2,
        repeats=1,
        device=torch.device("cpu"),
    )
    assert target.model.training is False
