"""Tests for ``_resolve_stages``: routing, validation, dedup, priority order.

Plus ``EfficiencySession.run`` stage-failure isolation: one stage raising
(e.g. CUDA OOM) is recorded in ``report.errors`` while later stages still run.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from utils.config.run_config import RunConfig
from utils.core import EFFICIENCY_STAGES, register_efficiency_stage
from utils.efficiency.session import EfficiencySession, _resolve_stages
from utils.efficiency.stages.base import BenchmarkTarget, EfficiencyStage, StageContext


class _FakeStage(EfficiencyStage):
    """Concrete stage double; priority is set per registered subclass."""

    def run(self, ctx: StageContext) -> Any:
        return None

    @classmethod
    def format_table(cls, result: Any) -> None:
        return None


@pytest.fixture
def two_fake_stages(registry_snapshot: None) -> None:
    """Register two same-priority stages under known registry keys."""
    for name in ("utest_stage_b", "utest_stage_a"):

        @register_efficiency_stage(name)
        class _Stage(_FakeStage):
            pass

        _Stage.name = name
        _Stage.priority = 50


class TestResolveStages:
    def test_empty_modes_selects_all_stages_by_priority(self) -> None:
        stages = _resolve_stages([])
        names = [name for name, _ in stages]
        from utils.core import get_supported_stages

        assert set(names) == set(get_supported_stages())
        priorities = [stage.priority for _, stage in stages]
        assert priorities == sorted(priorities)
        assert names == ["profile", "inference", "windowlate", "train", "trace"]

    def test_unknown_mode_exits_with_available_list(self) -> None:
        with pytest.raises(SystemExit, match="Unknown efficiency stage"):
            _resolve_stages(["profile", "utest_no_such_stage"])

    def test_selection_returns_fresh_stage_instances(self) -> None:
        stages = _resolve_stages(["train"])
        assert [name for name, _ in stages] == ["train"]
        again = _resolve_stages(["train"])
        assert again[0][1] is not stages[0][1]

    def test_dedup_preserves_first_seen_order(self, two_fake_stages: None) -> None:
        stages = _resolve_stages(["utest_stage_b", "utest_stage_a", "utest_stage_b"])
        assert [name for name, _ in stages] == ["utest_stage_b", "utest_stage_a"]

    def test_priority_sort_is_stable_within_equal_priority(
        self, two_fake_stages: None
    ) -> None:
        # Both fake stages share priority 50: first-seen order survives the sort.
        stages = _resolve_stages(["utest_stage_a", "utest_stage_b"])
        assert [name for name, _ in stages] == ["utest_stage_a", "utest_stage_b"]

    def test_registry_key_used_as_result_key(self, two_fake_stages: None) -> None:
        # Keys come from the registry name, not the stage's own ClassVar.
        stages = _resolve_stages(["utest_stage_a"])
        assert [name for name, _ in stages] == ["utest_stage_a"]
        assert EFFICIENCY_STAGES.get("utest_stage_a") is not None


class _FakeTarget:
    """Duck-typed BenchmarkTarget: CPU device, one synthetic batch."""

    device = torch.device("cpu")
    model = torch.nn.Linear(2, 2)

    def prepare(self, device: torch.device) -> None:
        pass

    @property
    def train_data(self) -> list[dict[str, Any]]:
        return [{"questions": torch.zeros(2, 3, dtype=torch.long)}]

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"y_label": torch.zeros(6)}


def _make_session(
    tmp_path: Path, modes: str, target: _FakeTarget | None = None
) -> EfficiencySession:
    # eff_cfg must be a real dataclass: session serialization runs it through
    # ``config_to_dict`` → ``asdict``, which rejects SimpleNamespace.
    @dataclass
    class _GeneralCfg:
        # default_factory closure: a plain ``= modes`` default would shadow the
        # enclosing parameter inside the class body.
        modes: str = field(default_factory=lambda: modes)
        resource_sample_interval: float = 0.05

    @dataclass
    class _EffCfg:
        general: _GeneralCfg = field(default_factory=_GeneralCfg)

    rc = SimpleNamespace(
        experiment=SimpleNamespace(model_name="UTestModel"),
        data=SimpleNamespace(dataset="tinyds", max_seq_len=3),
        general=SimpleNamespace(seed=42),
    )
    # ``rc`` is a SimpleNamespace mirroring the RunConfig read-surface and the
    # stub target is a partial double: cast marks the typed session boundary.
    return EfficiencySession(
        target=cast(BenchmarkTarget, target or _FakeTarget()),
        rc=cast(RunConfig, rc),
        eff_cfg=_EffCfg(),
        output_dir=tmp_path,
    )


@pytest.fixture
def fail_then_ok_stages(registry_snapshot: None) -> None:
    """A stage that raises OOM (runs first) and a healthy stage (runs second)."""

    @register_efficiency_stage("utest_fail_stage")
    class _FailStage(_FakeStage):
        priority = 10

        def run(self, ctx: StageContext) -> None:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. (fake)")

    @register_efficiency_stage("utest_ok_stage")
    class _OkStage(_FakeStage):
        priority = 20

        def run(self, ctx: StageContext) -> dict[str, bool]:
            return {"ok": True}


class TestStageFailureIsolation:
    def test_failed_stage_recorded_and_later_stages_run(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        report = _make_session(tmp_path, "utest_fail_stage,utest_ok_stage").run()
        assert report.results == {"utest_ok_stage": {"ok": True}}
        assert "utest_fail_stage" in report.errors
        assert "OutOfMemoryError" in report.errors["utest_fail_stage"]
        assert report.modes == ["utest_fail_stage", "utest_ok_stage"]

    def test_errors_persist_to_report_json(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        _make_session(tmp_path, "utest_fail_stage,utest_ok_stage").run()
        payload = json.loads((tmp_path / "efficiency_report.json").read_text())
        assert "OutOfMemoryError" in payload["errors"]["utest_fail_stage"]
        assert payload["results"]["utest_ok_stage"] == {"ok": True}

    def test_all_stages_failed_raises_after_writing_report(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="All efficiency stages failed"):
            _make_session(tmp_path, "utest_fail_stage").run()
        payload = json.loads((tmp_path / "efficiency_report.json").read_text())
        assert set(payload["errors"]) == {"utest_fail_stage"}
        assert payload["results"] == {}

    def test_no_failure_leaves_errors_empty(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        report = _make_session(tmp_path, "utest_ok_stage").run()
        assert report.errors == {}
        assert report.results == {"utest_ok_stage": {"ok": True}}


@pytest.fixture
def ctx_capture_stage(registry_snapshot: None) -> dict[str, Any]:
    """A stage capturing the StageContext fields the session feeds it."""
    captured: dict[str, Any] = {}

    @register_efficiency_stage("utest_capture_stage")
    class _CaptureStage(_FakeStage):
        priority = 5

        def run(self, ctx: StageContext) -> dict[str, bool]:
            captured.update(
                valid_tokens=ctx.valid_tokens,
                total=ctx.valid_tokens_total,
                batches=ctx.valid_tokens_batches,
            )
            return {"ok": True}

    return captured


class TestSplitThroughputNumerator:
    """The throughput numerator must be a full-split mean with provenance."""

    def test_ctx_carries_split_mean_and_provenance(
        self, ctx_capture_stage: dict[str, Any], tmp_path: Path
    ) -> None:
        # _FakeTarget yields one batch whose forward scores 6 tokens: the split
        # mean equals the single batch's count, but arrives with provenance.
        report = _make_session(tmp_path, "utest_capture_stage").run()
        assert report.results == {"utest_capture_stage": {"ok": True}}
        assert ctx_capture_stage["valid_tokens"] == pytest.approx(6.0)
        assert ctx_capture_stage["total"] == 6
        assert ctx_capture_stage["batches"] == 1

    def test_mean_averages_over_all_batches(
        self, ctx_capture_stage: dict[str, Any], tmp_path: Path
    ) -> None:
        class _TwoBatchTarget(_FakeTarget):
            """Loader yields two batches scoring 6 and 3 valid tokens."""

            @property
            def train_data(self) -> list[dict[str, Any]]:
                return [
                    {
                        "n": 6,
                        "questions": torch.zeros(2, 3, dtype=torch.long),
                    },
                    {
                        "n": 3,
                        "questions": torch.zeros(2, 3, dtype=torch.long),
                    },
                ]

            def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
                return {"y_label": torch.zeros(batch["n"])}

        report = _make_session(
            tmp_path, "utest_capture_stage", target=_TwoBatchTarget()
        ).run()
        assert report.results == {"utest_capture_stage": {"ok": True}}
        assert ctx_capture_stage["valid_tokens"] == pytest.approx(4.5)  # (6+3)/2
        assert ctx_capture_stage["total"] == 9
        assert ctx_capture_stage["batches"] == 2

    def test_failing_split_pass_aborts_the_session(
        self, ctx_capture_stage: dict[str, Any], tmp_path: Path
    ) -> None:
        """A broken loader during the full pass is a setup failure, not a
        silent fallback: it must propagate and abort the run."""

        class _BoomOnSecondAccessTarget(_FakeTarget):
            """Prefetch works; the split pass's loader access raises."""

            def __init__(self) -> None:
                super().__init__()
                self._accesses = 0

            @property
            def train_data(self) -> list[dict[str, Any]]:
                self._accesses += 1
                if self._accesses > 1:
                    raise RuntimeError("split pass boom")
                return [{"questions": torch.zeros(2, 3, dtype=torch.long)}]

        target = _BoomOnSecondAccessTarget()
        with pytest.raises(RuntimeError, match="split pass boom"):
            _make_session(tmp_path, "utest_capture_stage", target=target).run()
