"""Tests for environment snapshot fields and stage-scoped resource sampling."""

import pytest
import torch

from utils.efficiency.environment.env_info import EnvironmentInfo, collect_environment
from utils.efficiency.environment.sampling import (
    RESOURCE_METRICS,
    ResourceSummary,
    StageScopedSampler,
    _summarize,
)


class TestEnvironmentInfo:
    def test_determinism_dict_keys_and_defaults(self) -> None:
        info = EnvironmentInfo()
        assert info.determinism_dict() == {
            "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "deterministic_algorithms": False,
        }

    def test_collect_environment_on_cpu(self) -> None:
        info = collect_environment(torch.device("cpu"), torch.nn.Linear(2, 1))
        assert info.device_type == "cpu"
        assert info.torch_version
        assert info.python_version
        assert info.platform
        assert set(info.determinism_dict()) == {
            "cudnn_benchmark",
            "cudnn_deterministic",
            "deterministic_algorithms",
        }

    def test_collect_environment_reads_model_dtype(self) -> None:
        info = collect_environment(torch.device("cpu"), torch.nn.Linear(2, 1))
        assert info.model_dtype == "torch.float32"

    def test_resource_metrics_cover_all_stats_fields(self) -> None:
        import dataclasses

        from utils.efficiency.environment.sampling import ResourceStats

        stats_fields = {f.name for f in dataclasses.fields(ResourceStats)}
        assert {m.stats_field for m in RESOURCE_METRICS} == stats_fields


class TestSummarize:
    def test_empty_returns_defaults(self) -> None:
        assert _summarize([]) == ResourceSummary()

    def test_known_values(self) -> None:
        summary = _summarize([1.0, 2.0, 3.0])
        assert summary.mean == pytest.approx(2.0)
        assert summary.peak == 3.0
        assert summary.min == 1.0
        assert summary.p50 == 2.0
        assert summary.n == 3

    def test_even_sample_median_is_mean(self) -> None:
        assert _summarize([1.0, 2.0]).p50 == 1.5


class TestStageScopedSampler:
    def test_routing_and_stop_aggregation_without_thread(self) -> None:
        sampler = StageScopedSampler(torch.device("cpu"), interval_s=1.0)
        sampler.begin_stage("a")
        sampler._route({"cpu": 5.0, "rss": 100.0})
        sampler.end_stage()
        # samples outside a stage are dropped
        sampler._route({"cpu": 99.0})
        sampler.begin_stage("b")
        sampler._route({"cpu": 7.0})
        sampler.stop()

        stats = sampler.stop()  # second stop: no buckets added, still safe
        assert set(stats) == {"a", "b"}
        assert stats["a"].cpu_percent.mean == 5.0
        assert stats["a"].cpu_percent.n == 1
        assert stats["a"].process_rss_mib.mean == 100.0
        assert stats["b"].cpu_percent.mean == 7.0
        # unsampled GPU metrics fall back to empty summaries
        assert stats["a"].gpu_util_pct.n == 0
        assert stats["a"].gpu_util_pct.mean is None

    def test_begin_stage_creates_bucket_only_once(self) -> None:
        sampler = StageScopedSampler(torch.device("cpu"), interval_s=1.0)
        sampler.begin_stage("x")
        sampler.begin_stage("x")
        sampler.end_stage()
        sampler._route({"cpu": 1.0})  # no current stage: dropped
        stats = sampler.stop()
        assert list(stats) == ["x"]
        assert stats["x"].cpu_percent.n == 0
