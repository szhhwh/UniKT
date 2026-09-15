"""Tests for optuna config: directions, space validation, samplers, pruners, yaml."""

import dataclasses
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from optuna.pruners import MedianPruner, PercentilePruner
from optuna.samplers import BaseSampler, GridSampler, RandomSampler

from utils.core import register_model_config
from utils.optuna_utils.config import (
    HyperparameterSpace,
    OptunaConfig,
    direction_for_metric,
    load_optuna_config,
    param_spaces_from_model_config,
)

# --- direction_for_metric ---


class TestDirectionForMetric:
    def test_known_metrics(self) -> None:
        assert direction_for_metric("auc") == "maximize"
        assert direction_for_metric("acc") == "maximize"
        assert direction_for_metric("auprc") == "maximize"
        assert direction_for_metric("rmse") == "minimize"
        assert direction_for_metric("loss") == "minimize"

    def test_case_insensitive(self) -> None:
        assert direction_for_metric("AUC") == "maximize"

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported metric 'f1'"):
            direction_for_metric("f1")


# --- HyperparameterSpace.validate ---


class TestHyperparameterSpaceValidate:
    def test_valid_spaces_pass(self) -> None:
        HyperparameterSpace(name="a", type="int", low=1, high=8).validate()
        HyperparameterSpace(
            name="b", type="float", low=1e-4, high=1e-1, log=True
        ).validate()
        HyperparameterSpace(name="c", type="categorical", choices=[1, 2]).validate()

    @pytest.mark.parametrize("ptype", ["int", "float"])
    def test_missing_bounds(self, ptype: str) -> None:
        with pytest.raises(ValueError, match="requires 'low' and 'high'"):
            HyperparameterSpace(name="x", type=ptype).validate()

    @pytest.mark.parametrize("ptype", ["int", "float"])
    def test_low_must_be_below_high(self, ptype: str) -> None:
        with pytest.raises(ValueError, match="low must be less than high"):
            HyperparameterSpace(name="x", type=ptype, low=8, high=8).validate()
        with pytest.raises(ValueError, match="low must be less than high"):
            HyperparameterSpace(name="x", type=ptype, low=9, high=8).validate()

    @pytest.mark.parametrize("ptype", ["int", "float"])
    def test_log_and_step_mutually_exclusive(self, ptype: str) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            HyperparameterSpace(
                name="x", type=ptype, low=1, high=8, log=True, step=2
            ).validate()

    @pytest.mark.parametrize("ptype", ["int", "float"])
    def test_step_must_be_positive(self, ptype: str) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            HyperparameterSpace(name="x", type=ptype, low=1, high=8, step=0).validate()

    def test_categorical_requires_choices(self) -> None:
        with pytest.raises(ValueError, match="requires 'choices'"):
            HyperparameterSpace(name="x", type="categorical").validate()

    def test_unsupported_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported parameter type: str"):
            HyperparameterSpace(name="x", type="str").validate()

    @pytest.mark.parametrize("ptype", ["int", "float"])
    def test_default_out_of_range(self, ptype: str) -> None:
        with pytest.raises(ValueError, match="out of range"):
            HyperparameterSpace(
                name="x", type=ptype, low=1, high=8, default=16
            ).validate()

    def test_default_not_in_choices(self) -> None:
        with pytest.raises(ValueError, match="not in choices"):
            HyperparameterSpace(
                name="x", type="categorical", choices=[1, 2], default=3
            ).validate()

    def test_default_inside_range_passes(self) -> None:
        HyperparameterSpace(name="x", type="int", low=1, high=8, default=4).validate()


# --- suggest via FixedTrial ---


class TestSuggest:
    def test_int_float_categorical_return_exact_values(self, fixed_trial: Any) -> None:
        trial = fixed_trial({"depth": 4, "lr": 0.01, "act": "relu"})
        assert (
            HyperparameterSpace(name="depth", type="int", low=1, high=8).suggest(trial)
            == 4
        )
        assert HyperparameterSpace(name="lr", type="float", low=1e-4, high=1.0).suggest(
            trial
        ) == pytest.approx(0.01)
        assert (
            HyperparameterSpace(
                name="act", type="categorical", choices=["relu", "tanh"]
            ).suggest(trial)
            == "relu"
        )

    def test_int_step_passed_through(self, fixed_trial: Any) -> None:
        trial = fixed_trial({"depth": 4})
        got = HyperparameterSpace(
            name="depth", type="int", low=0, high=8, step=2
        ).suggest(trial)
        assert got == 4

    def test_invalid_space_refused_before_suggesting(self, fixed_trial: Any) -> None:
        trial = fixed_trial({"depth": 4})
        with pytest.raises(ValueError, match="low must be less than high"):
            HyperparameterSpace(name="depth", type="int", low=9, high=2).suggest(trial)


# --- OptunaConfig.get_sampler / get_pruner ---


def _first_suggestion(sampler: BaseSampler) -> float:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    return study.trials[0].params["x"]


class TestGetSampler:
    def test_seed_injected_makes_runs_reproducible(self) -> None:
        cfg = OptunaConfig(sampler="random", seed=13)
        assert _first_suggestion(cfg.get_sampler()) == _first_suggestion(
            OptunaConfig(sampler="random", seed=13).get_sampler()
        )

    def test_different_seeds_differ(self) -> None:
        a = _first_suggestion(OptunaConfig(sampler="random", seed=13).get_sampler())
        b = _first_suggestion(OptunaConfig(sampler="random", seed=99).get_sampler())
        assert a != b

    def test_explicit_sampler_kwargs_seed_not_overridden(self) -> None:
        cfg = OptunaConfig(sampler="random", seed=1, sampler_kwargs={"seed": 99})
        assert _first_suggestion(cfg.get_sampler()) == _first_suggestion(
            OptunaConfig(sampler="random", sampler_kwargs={"seed": 99}).get_sampler()
        )

    def test_random_sampler_type(self) -> None:
        assert isinstance(OptunaConfig(sampler="random").get_sampler(), RandomSampler)

    def test_grid_requires_search_space(self) -> None:
        with pytest.raises(ValueError, match="GridSampler requires 'search_space'"):
            OptunaConfig(sampler="grid").get_sampler()

    def test_grid_with_search_space(self) -> None:
        cfg = OptunaConfig(
            sampler="grid", sampler_kwargs={"search_space": {"x": [1, 2]}}
        )
        assert isinstance(cfg.get_sampler(), GridSampler)

    def test_unknown_sampler(self) -> None:
        with pytest.raises(ValueError, match="Unsupported sampler: magic"):
            OptunaConfig(sampler="magic").get_sampler()


class TestGetPruner:
    def test_none_pruer_disabled(self) -> None:
        # The pruner field is documented None-able but annotated ``str`` in the
        # config; keep the value Any to exercise the documented runtime path.
        pruner: Any = None
        assert OptunaConfig(pruner=pruner).get_pruner() is None

    def test_median_default(self) -> None:
        assert isinstance(OptunaConfig().get_pruner(), MedianPruner)

    def test_percentile_requires_kwarg(self) -> None:
        with pytest.raises(ValueError, match="requires 'percentile'"):
            OptunaConfig(pruner="percentile").get_pruner()

    def test_percentile_with_kwarg(self) -> None:
        cfg = OptunaConfig(pruner="percentile", pruner_kwargs={"percentile": 25.0})
        assert isinstance(cfg.get_pruner(), PercentilePruner)

    def test_unknown_pruner(self) -> None:
        with pytest.raises(ValueError, match="Unsupported pruner: magic"):
            OptunaConfig(pruner="magic").get_pruner()


# --- load_optuna_config ---


class TestLoadOptunaConfig:
    def test_yaml_round_trip(self, write_optuna_yaml: Callable[..., str]) -> None:
        path = write_optuna_yaml(
            {
                "sampler": "random",
                "seed": 7,
                "n_trials": 3,
                "directions": ["maximize", "minimize"],
                "pruner_kwargs": {"n_startup_trials": 2},
            }
        )
        cfg = load_optuna_config(path)
        assert cfg.sampler == "random"
        assert cfg.seed == 7
        assert cfg.n_trials == 3
        assert cfg.directions == ["maximize", "minimize"]
        assert cfg.pruner_kwargs == {"n_startup_trials": 2}

    def test_unknown_key_raises_type_error(
        self, write_optuna_yaml: Callable[..., str]
    ) -> None:
        path = write_optuna_yaml({"no_such_knob": 1})
        with pytest.raises(TypeError, match="no_such_knob"):
            load_optuna_config(path)

    def test_empty_yaml_gives_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_optuna_config(str(path)) == OptunaConfig()


# --- param_spaces_from_model_config ---


class TestParamSpacesFromModelConfig:
    def test_spaces_derived_from_field_metadata(self, registry_snapshot: None) -> None:
        @register_model_config("UTestOptunaModel")
        class UTestOptunaModel:
            epochs: int = dataclasses.field(
                default=10, metadata={"optuna": {"type": "int", "low": 5, "high": 20}}
            )
            hidden: int = dataclasses.field(
                default=64, metadata={"optuna": {"type": "int", "low": 32, "high": 128}}
            )
            act: str = dataclasses.field(
                default="tanh",
                metadata={
                    "optuna": {"type": "categorical", "choices": ["relu", "tanh"]}
                },
            )
            plain: int = 7  # no metadata: skipped

        spaces = param_spaces_from_model_config("UTestOptunaModel")
        by_name = {s.name: s for s in spaces}
        assert set(by_name) == {"epochs", "hidden", "act"}
        assert by_name["epochs"].low == 5 and by_name["epochs"].high == 20
        assert by_name["epochs"].default == 10
        assert by_name["act"].choices == ["relu", "tanh"]

    def test_out_of_range_default_dropped(self, registry_snapshot: None) -> None:
        @register_model_config("UTestOptunaBadDefault")
        class UTestOptunaBadDefault:
            hidden: int = dataclasses.field(
                default=512,
                metadata={"optuna": {"type": "int", "low": 32, "high": 128}},
            )

        spaces = param_spaces_from_model_config("UTestOptunaBadDefault")
        assert len(spaces) == 1
        assert spaces[0].default is None

    def test_unregistered_model_raises_key_error(self, registry_snapshot: None) -> None:
        with pytest.raises(KeyError, match=re.escape("UTestNoSuchModel")):
            param_spaces_from_model_config("UTestNoSuchModel")
