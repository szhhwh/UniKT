"""Tests for stage helpers: kv tables, training benchmark, profile, operator stats."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rich.table import Table

from utils.efficiency.stages.base import EfficiencyStage, StageContext
from utils.efficiency.stages.profile import profile_model
from utils.efficiency.stages.trace import _aggregate, _extract_operators
from utils.efficiency.stages.training import benchmark_training
from utils.efficiency.target import BenchmarkTarget

# --- base: table helpers ---


class _TableOwner(EfficiencyStage):
    """Minimal concrete subclass exposing the shared table helpers."""

    def run(self, ctx: StageContext) -> None:
        return None

    @classmethod
    def format_table(cls, result: Any) -> None:
        return None


def _latency_result(**overrides: Any) -> SimpleNamespace:
    base = {
        "latency_mean_ms": 1.0,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 2.0,
        "latency_p99_ms": 3.0,
        "latency_cv": 0.1,
        "latency_repeat_cv": 0.2,
        "gpu_peak_allocated_mib": None,
        "gpu_peak_reserved_mib": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestMakeKvTable:
    def test_shape_and_style(self) -> None:
        table = _TableOwner.make_kv_table("Unit Test Title")
        assert isinstance(table, Table)
        assert table.title == "Unit Test Title"
        assert table.show_header is False
        assert len(table.columns) == 2

    def test_starts_empty(self) -> None:
        assert len(_TableOwner.make_kv_table("t").rows) == 0


class TestAddLatencyRows:
    def test_base_rows_without_gpu_peaks(self) -> None:
        table = _TableOwner.make_kv_table("t")
        EfficiencyStage.add_latency_rows(table, _latency_result())
        assert len(table.rows) == 4  # mean, p50/p95/p99, cv, repeat cv

    def test_gpu_peak_rows_added_when_present(self) -> None:
        table = _TableOwner.make_kv_table("t")
        EfficiencyStage.add_latency_rows(
            table,
            _latency_result(gpu_peak_allocated_mib=12.0, gpu_peak_reserved_mib=16.0),
        )
        assert len(table.rows) == 6

    def test_only_one_gpu_peak_present_adds_one_row(self) -> None:
        table = _TableOwner.make_kv_table("t")
        EfficiencyStage.add_latency_rows(
            table, _latency_result(gpu_peak_reserved_mib=16.0)
        )
        assert len(table.rows) == 5


# --- training stage: real CPU optimizer steps ---


class _TrainTarget:
    """Duck-typed BenchmarkTarget whose train step is a real SGD update."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(3, 2)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.steps = 0

    def compute_train_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[dict[str, Any], torch.Tensor]:
        x, y = batch
        self.opt.zero_grad()
        loss = torch.nn.functional.mse_loss(self.model(x), y)
        loss.backward()
        self.opt.step()
        self.steps += 1
        return {}, loss.detach()

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {"y_hat": self.model(batch[0])}


_BATCH = (torch.randn(4, 3), torch.randn(4, 2))


class TestBenchmarkTraining:
    def test_runs_real_steps_and_forces_train_mode(self) -> None:
        target = _TrainTarget()
        target.model.eval()
        metrics = benchmark_training(
            cast(BenchmarkTarget, target),
            _BATCH,
            batch_size=4,
            valid_tokens=8,
            valid_tokens_total=16,
            valid_tokens_batches=2,
            warmup_iters=1,
            iters=2,
            device=torch.device("cpu"),
        )
        assert target.model.training is True
        # warmup + timed loop each run a full optimizer step
        assert target.steps == 3
        assert metrics.iters == 2
        assert metrics.batch_size == 4
        assert metrics.valid_tokens_per_batch == 8
        assert metrics.valid_tokens_total == 16
        assert metrics.valid_tokens_batches == 2
        assert metrics.wall_time_s > 0
        assert metrics.gpu_peak_allocated_mib is None

    def test_throughput_formula_sanity(self) -> None:
        target = _TrainTarget()
        metrics = benchmark_training(
            cast(BenchmarkTarget, target),
            _BATCH,
            batch_size=4,
            valid_tokens=8,
            valid_tokens_total=16,
            valid_tokens_batches=2,
            warmup_iters=1,
            iters=2,
            device=torch.device("cpu"),
        )
        expected_throughput = 8 * 2 / metrics.wall_time_s
        assert metrics.throughput_interactions_per_sec == pytest.approx(
            expected_throughput
        )
        assert metrics.samples_per_sec == pytest.approx(4 * 2 / metrics.wall_time_s)
        # ns_per_interaction and throughput are exact reciprocals
        assert (
            metrics.throughput_interactions_per_sec * metrics.ns_per_interaction
            == pytest.approx(1e9, rel=1e-6)
        )

    def test_weights_actually_update(self) -> None:
        target = _TrainTarget()
        before = target.model.weight.detach().clone()
        benchmark_training(
            cast(BenchmarkTarget, target),
            _BATCH,
            batch_size=4,
            valid_tokens=8,
            valid_tokens_total=8,
            valid_tokens_batches=1,
            warmup_iters=0,
            iters=1,
            device=torch.device("cpu"),
        )
        assert not torch.equal(before, target.model.weight.detach())


# --- profile stage ---


class TestProfileModel:
    def test_small_linear_profile(self) -> None:
        model = torch.nn.Linear(3, 2)
        forward = lambda: model(torch.randn(4, 3))  # noqa: E731
        profile = profile_model(model, forward, torch.device("cpu"))
        assert profile.params == 8
        assert profile.trainable_params == 8
        # size is rounded to 3 decimals, so it is 0.0 for this tiny model
        assert profile.model_size_mb == pytest.approx(8 * 4 / 1024**2, abs=1e-3)
        assert profile.flops_forward == 48
        assert profile.op_breakdown
        values = list(profile.op_breakdown.values())
        assert values == sorted(values, reverse=True)

    def test_frozen_params_counted_separately(self) -> None:
        model = torch.nn.Linear(3, 2)
        model.weight.requires_grad_(False)
        profile = profile_model(model, lambda: None, torch.device("cpu"))
        assert profile.params == 8
        assert profile.trainable_params == 2

    def test_count_flops_disabled(self) -> None:
        model = torch.nn.Linear(3, 2)
        profile = profile_model(
            model,
            lambda: model(torch.randn(2, 3)),
            torch.device("cpu"),
            count_flops=False,
        )
        assert profile.flops_forward is None
        assert profile.op_breakdown == {}


# --- trace stage: hand-made fake profiler events ---


def _event(
    key: str,
    *,
    count: int = 1,
    cpu_total: float = 0.0,
    self_cpu: float = 0.0,
    device_total: float = 0.0,
    self_device: float = 0.0,
    flops: int = 0,
    self_cpu_mem: int = 0,
    self_device_mem: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        key=key,
        count=count,
        cpu_time_total=cpu_total,
        self_cpu_time_total=self_cpu,
        device_time_total=device_total,
        self_device_time_total=self_device,
        flops=flops,
        self_cpu_memory_usage=self_cpu_mem,
        self_device_memory_usage=self_device_mem,
    )


class TestExtractOperators:
    def test_cpu_sort_by_self_cpu_and_topn(self) -> None:
        events = [
            _event("aten::add", self_cpu=5.0),
            _event("aten::mm", self_cpu=20.0),
            _event("aten::relu", self_cpu=10.0),
        ]
        ops = _extract_operators(events, top_ops=2, device=torch.device("cpu"))
        assert [op.name for op in ops] == ["aten::mm", "aten::relu"]
        assert ops[0].self_cpu_us == 20.0
        # CUDA-side fields stay zeroed on a CPU device
        assert ops[0].cuda_total_us == 0.0
        assert ops[0].self_cuda_us == 0.0
        assert ops[0].self_cuda_mem_bytes == 0

    def test_calls_and_flops_copied(self) -> None:
        events = [_event("aten::mm", count=3, self_cpu=1.0, flops=48)]
        ops = _extract_operators(events, top_ops=5, device=torch.device("cpu"))
        assert ops[0].calls == 3
        assert ops[0].flops == 48
        assert ops[0].self_cpu_mem_bytes == 0


class TestAggregate:
    def test_cpu_sums_all_events(self) -> None:
        events = [
            _event("a", self_cpu=1.5, flops=10),
            _event("b", self_cpu=2.5, flops=30),
        ]
        total_cpu, total_cuda, op_count, total_flops = _aggregate(
            events, torch.device("cpu")
        )
        assert total_cpu == 4.0
        assert total_cuda == 0.0
        assert op_count == 2
        assert total_flops == 40

    def test_cuda_sums_only_pure_kernel_events(self) -> None:
        # Host op mirrors its child kernel's device time; only events with
        # cpu_time_total == 0 are pure kernels and must be the sole contributors.
        host_op = _event("aten::mm", cpu_total=10.0, self_device=7.0)
        kernel = _event("void mm_kernel", cpu_total=0.0, self_device=7.0)
        _, total_cuda, _, _ = _aggregate([host_op, kernel], torch.device("cuda"))
        assert total_cuda == 7.0

    def test_missing_attributes_tolerated(self) -> None:
        blank = SimpleNamespace(key="blank")
        total_cpu, total_cuda, op_count, total_flops = _aggregate(
            [blank], torch.device("cpu")
        )
        assert (total_cpu, total_cuda, op_count, total_flops) == (0.0, 0.0, 1, 0)
