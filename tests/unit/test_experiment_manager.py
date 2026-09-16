"""Tests for ExperimentManager: directory layout, collision handling, factories."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from utils.config.run_config import RunConfig

from utils.experiment_manager import ExperimentManager, ExperimentType


def _make(base_dir: Path, **kwargs: Any) -> ExperimentManager:
    defaults: dict[str, Any] = {
        "exp_type": ExperimentType.NORMAL,
        "model_name": "TinyModel",
        "dataset_name": "tinyds",
        "base_dir": str(base_dir),
    }
    defaults.update(kwargs)
    return ExperimentManager(**defaults)


# --- construction ---


class TestInit:
    def test_creates_timestamped_dir_immediately(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path)
        assert mgr.exp_dir.exists()
        assert mgr.exp_dir.parent == tmp_path / "normal"
        name = mgr.exp_dir.name
        assert name.startswith("TinyModel_tinyds_")
        assert name[len("TinyModel_tinyds_") :].count("-") == 1  # ts suffix

    def test_tags_join_into_dir_name(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path, tags=["fold0", "bs32"])
        assert mgr.exp_dir.name.endswith("_fold0_bs32")

    def test_no_tags_no_trailing_underscore(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path)
        assert not mgr.exp_dir.name.endswith("_")

    def test_exp_type_selects_subdir(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path, exp_type=ExperimentType.HYPERPARAM_SEARCH)
        assert mgr.exp_dir.parent == tmp_path / "hyperparam_search"

    def test_get_log_dir_returns_str_of_exp_dir(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path)
        assert isinstance(mgr.get_log_dir(), str)
        assert mgr.get_log_dir() == str(mgr.exp_dir)


# --- collision handling ---


class TestUniqueDir:
    def test_collision_appends_suffix(self, tmp_path: Path) -> None:
        parent = tmp_path / "normal"
        (parent / "Name").mkdir(parents=True)
        result = ExperimentManager._create_unique_dir(parent, "Name")
        assert result == parent / "Name_2"
        assert result.exists()

    def test_third_collision_appends_3(self, tmp_path: Path) -> None:
        parent = tmp_path / "normal"
        (parent / "Name").mkdir(parents=True)
        (parent / "Name_2").mkdir()
        result = ExperimentManager._create_unique_dir(parent, "Name")
        assert result == parent / "Name_3"

    def test_two_managers_same_second_disambiguate(self, tmp_path: Path) -> None:
        first = _make(tmp_path)
        second = _make(tmp_path)
        assert first.exp_dir != second.exp_dir
        assert first.exp_dir.exists() and second.exp_dir.exists()


# --- sub experiments ---


class TestSubExperiment:
    def test_shares_parent_timestamp_no_new_ts_dir(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path)
        normal_before = set((tmp_path / "normal").iterdir())
        sub = mgr.create_sub_experiment("trial_0")
        assert sub.exp_dir == mgr.exp_dir / "trial_0"
        assert sub.exp_dir.exists()
        assert set((tmp_path / "normal").iterdir()) == normal_before

    def test_child_inherits_identity_and_appends_tag(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path, tags=["fold1"])
        sub = mgr.create_sub_experiment("trial_2")
        assert sub.tags == ["fold1", "trial_2"]
        assert sub.model_name == mgr.model_name
        assert sub.is_existing_run == mgr.is_existing_run

    def test_create_subdir_returns_path(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path)
        sub = mgr.create_subdir("figures")
        assert sub == mgr.exp_dir / "figures"
        assert sub.exists()


# --- factories ---


class TestFactories:
    def test_from_args_extracts_tags(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            model="M", dataset="D", fold=3, batch_size=64, base_dir=str(tmp_path)
        )
        mgr = ExperimentManager.from_args(args, ExperimentType.NORMAL)
        assert mgr.exp_dir.name.endswith("_fold3_bs64")

    def test_from_args_tolerates_missing_optionals(self, tmp_path: Path) -> None:
        args = argparse.Namespace(model="M", dataset="D", base_dir=str(tmp_path))
        mgr = ExperimentManager.from_args(args, ExperimentType.NORMAL)
        assert mgr.tags == []

    def test_from_run_config_uses_runs_base(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        make_run_config: Callable[..., RunConfig],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        rc = make_run_config()
        mgr = ExperimentManager.from_run_config(rc, ExperimentType.NORMAL)
        # base_dir is hardcoded relative "runs"; the dir lands under the CWD.
        assert mgr.base_dir == Path("runs")
        assert mgr.exp_dir.parent == Path("runs") / "normal"
        assert (tmp_path / mgr.exp_dir).exists()
        assert mgr.exp_dir.name.endswith(f"_fold0_bs{rc.model.batch_size}")

    def test_from_run_dir_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            ExperimentManager.from_run_dir(tmp_path / "nope")

    def test_from_run_dir_binds_existing(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "normal" / "existing_run"
        run_dir.mkdir(parents=True)
        mgr = ExperimentManager.from_run_dir(run_dir)
        assert mgr.is_existing_run is True
        assert mgr.get_log_dir() == str(run_dir.resolve())
        # binding creates nothing new
        assert set(run_dir.iterdir()) == set()

    def test_from_run_dir_flag_inherited_by_sub(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "normal" / "existing_run"
        run_dir.mkdir(parents=True)
        mgr = ExperimentManager.from_run_dir(run_dir)
        sub = mgr.create_sub_experiment("sub")
        assert sub.is_existing_run is True


# --- info ---


class TestExperimentInfo:
    def test_info_shape_and_enum_value(self, tmp_path: Path) -> None:
        mgr = _make(tmp_path, tags=["t1"])
        info = mgr.get_experiment_info()
        assert info["experiment_type"] == "normal"  # enum value, not member
        assert info["model_name"] == "TinyModel"
        assert info["dataset_name"] == "tinyds"
        assert info["tags"] == ["t1"]
        assert info["experiment_dir"] == str(mgr.exp_dir)
