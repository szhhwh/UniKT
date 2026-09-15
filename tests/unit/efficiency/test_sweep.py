"""Tests for the sweep machinery: parsing, mutators, cartesian join, index IO."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from utils.efficiency.sweep import (
    EfficiencySweep,
    SweepFailure,
    SweepPoint,
    SweepReport,
    SweepRun,
    _apply_compile_state,
    _parse_batch_sizes,
    _parse_compile_modes,
    batch_size_sweep,
    cartesian_sweep,
    compile_sweep,
)


def _make_rc() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(batch_size=4),
        compile=SimpleNamespace(compile=False, compile_mode="default"),
    )


def _make_eff_cfg(batch_sizes: str = "", compile_modes: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        general=SimpleNamespace(batch_sizes=batch_sizes, compile_modes=compile_modes)
    )


class TestParseBatchSizes:
    def test_parses_strips_and_dedups_preserving_order(self) -> None:
        assert _parse_batch_sizes("8, 16,8, 32") == [8, 16, 32]

    def test_empty_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit, match="batch_sizes is empty"):
            _parse_batch_sizes("  ")

    def test_non_int_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit, match="invalid batch_size 'big'"):
            _parse_batch_sizes("8, big")

    def test_non_positive_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit, match="must be > 0, got 0"):
            _parse_batch_sizes("0")
        with pytest.raises(SystemExit, match="must be > 0, got -8"):
            _parse_batch_sizes("-8")


class TestParseCompileModes:
    def test_parses_and_dedups(self) -> None:
        assert _parse_compile_modes("off, default,off") == ["off", "default"]

    def test_empty_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit, match="compile_modes is empty"):
            _parse_compile_modes(",")

    def test_invalid_mode_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit, match="invalid compile mode 'turbo'"):
            _parse_compile_modes("turbo")


class TestMutators:
    def test_batch_size_sweep_labels_and_mutation(self) -> None:
        points = batch_size_sweep([8, 16])
        assert [p.label for p in points] == ["bs8", "bs16"]
        rc = _make_rc()
        for point in points:
            point.mutate(rc, None)
        assert rc.model.batch_size == 16

    def test_each_point_mutates_independently(self) -> None:
        rc = _make_rc()
        batch_size_sweep([8, 16])[0].mutate(rc, None)
        assert rc.model.batch_size == 8

    def test_compile_sweep_off_disables(self) -> None:
        rc = _make_rc()
        rc.compile.compile = True
        points = compile_sweep(["off"])
        assert points[0].label == "cmp_off"
        points[0].mutate(rc, None)
        assert rc.compile.compile is False

    def test_compile_sweep_mode_enables_and_sets_mode(self) -> None:
        rc = _make_rc()
        compile_sweep(["max-autotune"])[0].mutate(rc, None)
        assert rc.compile.compile is True
        assert rc.compile.compile_mode == "max-autotune"

    def test_apply_compile_state_direct(self) -> None:
        # SimpleNamespace mirrors the CompileConfig surface; cast marks the
        # typed boundary of the internal helper.
        cc: Any = SimpleNamespace(compile=True, compile_mode="default")
        _apply_compile_state(cc, "off")
        assert cc.compile is False
        _apply_compile_state(cc, "reduce-overhead")
        assert cc.compile is True
        assert cc.compile_mode == "reduce-overhead"


class TestCartesianSweep:
    def test_labels_join_and_both_mutators_apply(self) -> None:
        rc = _make_rc()
        points = cartesian_sweep(batch_size_sweep([32]), compile_sweep(["off"]))
        assert [p.label for p in points] == ["bs32_cmp_off"]
        points[0].mutate(rc, None)
        assert rc.model.batch_size == 32
        assert rc.compile.compile is False

    def test_full_product_shape(self) -> None:
        points = cartesian_sweep(
            batch_size_sweep([8, 16]), compile_sweep(["off", "default"])
        )
        assert [p.label for p in points] == [
            "bs8_cmp_off",
            "bs8_cmp_default",
            "bs16_cmp_off",
            "bs16_cmp_default",
        ]


class TestResolvePoints:
    def test_explicit_points_win(self) -> None:
        explicit = [SweepPoint("custom", lambda rc, cfg: None)]
        got = EfficiencySweep._resolve_points(explicit, _make_eff_cfg("8"))
        assert got is explicit

    def test_both_axes_make_cartesian(self) -> None:
        points = EfficiencySweep._resolve_points(
            None, _make_eff_cfg("8,16", "off,default")
        )
        assert [p.label for p in points] == [
            "bs8_cmp_off",
            "bs8_cmp_default",
            "bs16_cmp_off",
            "bs16_cmp_default",
        ]

    def test_compile_only_axis(self) -> None:
        points = EfficiencySweep._resolve_points(None, _make_eff_cfg("", "off"))
        assert [p.label for p in points] == ["cmp_off"]

    def test_batch_sizes_only_axis(self) -> None:
        points = EfficiencySweep._resolve_points(None, _make_eff_cfg("8,8", ""))
        assert [p.label for p in points] == ["bs8"]


class TestSweepReportIO:
    def test_write_json_round_trip(self, tmp_path: Path) -> None:
        report = SweepReport(
            model_name="GIKT",
            dataset_name="assist09",
            timestamp="2026-08-19 00:00:00",
            labels=["bs8", "bs16"],
            modes=["profile", "train"],
            config={"general": {"warmup_iters": 50}},
            sweep_dir=str(tmp_path / "sweep"),
            runs=[
                SweepRun("bs8", str(tmp_path / "bs8")),
                SweepRun("bs16", str(tmp_path / "bs16")),
            ],
        )
        out = tmp_path / "nested" / "sweep_index.json"
        report.write_json(out)
        payload = json.loads(out.read_text())
        assert payload["model_name"] == "GIKT"
        assert payload["labels"] == ["bs8", "bs16"]
        assert payload["runs"] == [
            {"label": "bs8", "dir": str(tmp_path / "bs8")},
            {"label": "bs16", "dir": str(tmp_path / "bs16")},
        ]
        assert payload["sweep_dir"] == str(tmp_path / "sweep")

    def test_failures_written_and_default_empty(self, tmp_path: Path) -> None:
        report = SweepReport(
            model_name="GIKT",
            dataset_name="assist09",
            timestamp="2026-08-19 00:00:00",
            labels=["bs8"],
            modes=["profile"],
            config={},
            sweep_dir=str(tmp_path / "sweep"),
            runs=[SweepRun("bs8", str(tmp_path / "bs8"))],
        )
        out = tmp_path / "sweep_index.json"
        report.write_json(out)
        assert json.loads(out.read_text())["failures"] == []

        report.failures.append(SweepFailure("bs999", "OutOfMemoryError: CUDA OOM"))
        report.write_json(out)
        assert json.loads(out.read_text())["failures"] == [
            {"label": "bs999", "error": "OutOfMemoryError: CUDA OOM"}
        ]
