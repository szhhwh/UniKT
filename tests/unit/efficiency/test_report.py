"""Tests for EfficiencyReport serialization and the resource-summary formatter."""

import json
from pathlib import Path

import pytest

from utils.efficiency.environment import EnvironmentInfo, ResourceSummary
from utils.efficiency.report import EfficiencyReport, _rs
from utils.efficiency.stages.training import TrainingMetrics


def _make_report() -> EfficiencyReport:
    return EfficiencyReport(
        model_name="GIKT",
        dataset_name="assist09",
        timestamp="2026-08-19 00:00:00",
        batch_size=32,
        seq_len=100,
        modes=["train"],
        config={"general": {"warmup_iters": 50}},
        determinism={"seed": 42},
        environment=EnvironmentInfo(device_type="cpu"),
        resource=None,
        results={"train": TrainingMetrics(iters=2, batch_size=32)},
    )


class TestEfficiencyReport:
    def test_to_dict_includes_nested_stage_results(self) -> None:
        payload = _make_report().to_dict()
        assert payload["model_name"] == "GIKT"
        assert payload["results"]["train"]["iters"] == 2
        assert payload["environment"]["device_type"] == "cpu"

    def test_write_json_round_trip(self, tmp_path: Path) -> None:
        report = _make_report()
        out = tmp_path / "sub" / "efficiency_report.json"
        report.write_json(out)
        payload = json.loads(out.read_text())
        assert payload["model_name"] == "GIKT"
        assert payload["results"]["train"]["iters"] == 2
        assert payload["determinism"] == {"seed": 42}


class TestStageErrors:
    def test_errors_serialize_to_json(self, tmp_path: Path) -> None:
        report = _make_report()
        report.errors = {"train": "RuntimeError: boom"}
        out = tmp_path / "efficiency_report.json"
        report.write_json(out)
        assert json.loads(out.read_text())["errors"] == {"train": "RuntimeError: boom"}

    def test_to_dict_defaults_to_empty_errors(self) -> None:
        assert _make_report().to_dict()["errors"] == {}

    def test_print_console_renders_failed_stage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = _make_report()
        report.errors = {"train": "RuntimeError: boom"}
        report.print_console()
        output = capsys.readouterr().out
        assert "Stage failed — train" in output
        assert "RuntimeError: boom" in output

    def test_error_table_collapses_and_truncates(self) -> None:
        from rich.console import Console

        from utils.efficiency.report import (
            _CONSOLE_ERROR_MAX_CHARS,
            _stage_error_table,
        )

        def render(message: str) -> str:
            console = Console(record=True, width=_CONSOLE_ERROR_MAX_CHARS + 10)
            console.print(_stage_error_table("train", message))
            return console.export_text()

        assert "a b c" in render("a\n  b\nc")

        long = "x" * (_CONSOLE_ERROR_MAX_CHARS + 50)
        text = render(long)
        assert "x" * (_CONSOLE_ERROR_MAX_CHARS - 1) in text
        assert "x" * _CONSOLE_ERROR_MAX_CHARS not in text  # truncated, not wrapped
        assert "…" in text


class TestRsHelper:
    def test_zero_samples_renders_dashes(self) -> None:
        assert _rs(ResourceSummary(n=0), "MiB") == ("—", "—")

    def test_values_render_with_unit(self) -> None:
        summary = ResourceSummary(mean=10.44, peak=20.46, n=3)
        assert _rs(summary, "MiB") == ("10.4 MiB", "20.5 MiB")

    def test_empty_unit_stripped(self) -> None:
        summary = ResourceSummary(mean=10.44, peak=20.46, n=3)
        assert _rs(summary, "") == ("10.4", "20.5")
