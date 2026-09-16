"""Tests for OptunaTuner: validation, in-memory search, result artifacts.

All studies are in-memory (``db_url`` unset); the two artifact tests point
``save_dir`` into ``tmp_path`` so the sqlite fallback stays local and temporary.
"""

from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml
from optuna.samplers import TPESampler

from utils.optuna_utils.config import HyperparameterSpace, OptunaConfig
from utils.optuna_utils.tuner import OptunaTuner

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _grid_cfg(**overrides: Any) -> OptunaConfig:
    base: dict[str, Any] = {
        "sampler": "grid",
        "sampler_kwargs": {"search_space": {"x": [1, 2]}},
        "pruner": None,
        "n_trials": 2,
        "verbose": 0,
        "study_name": "utest_study",
    }
    base.update(overrides)
    return OptunaConfig(**base)


def _x_space(
    default: int | None = None, low: int = 1, high: int = 2
) -> HyperparameterSpace:
    return HyperparameterSpace(
        name="x", type="int", low=low, high=high, default=default
    )


# --- construction ---


class TestConstruction:
    def test_invalid_space_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="low must be less than high"):
            OptunaTuner(
                _grid_cfg(), [_x_space(low=9, high=2)], lambda trial, params: 0.0
            )

    def test_valid_construction_leaves_study_unset(self) -> None:
        tuner = OptunaTuner(_grid_cfg(), [_x_space()], lambda trial, params: 0.0)
        assert tuner.study is None
        assert tuner._param_importances is None
        assert tuner._pareto_front is None


# --- search() end-to-end ---


class TestSearch:
    def test_grid_two_trials_deterministic(self) -> None:
        tuner = OptunaTuner(
            _grid_cfg(),
            [_x_space()],
            lambda trial, params: float(params["x"]),
        )
        best = tuner.search()
        assert best == {"x": 2}
        assert tuner.study is not None
        assert len(tuner.study.trials) == 2
        assert tuner._best_params() == {"x": 2}

    def test_study_name_persisted_onto_config(self) -> None:
        cfg = _grid_cfg(study_name=None)
        OptunaTuner(cfg, [_x_space()], lambda trial, params: 0.0).search()
        assert cfg.study_name is not None
        assert cfg.study_name.startswith("study_")

    def test_search_skips_default_enqueue_for_grid_sampler(self) -> None:
        # GridSampler exhausts the grid; a default-carrying space must not add
        # an extra enqueued startup trial (verified via the trial count).
        tuner = OptunaTuner(
            _grid_cfg(),
            [_x_space(default=2)],
            lambda trial, params: float(params["x"]),
        )
        tuner.search()
        assert tuner.study is not None
        assert len(tuner.study.trials) == 2

    def test_missing_direction_raises(self) -> None:
        cfg = _grid_cfg(directions=[])
        tuner = OptunaTuner(cfg, [_x_space()], lambda trial, params: 0.0)
        with pytest.raises(ValueError, match="directions list is empty"):
            tuner.search()


# --- _enqueue_defaults ---


class TestEnqueueDefaults:
    def _tuner(self, spaces: list[HyperparameterSpace]) -> OptunaTuner:
        tuner = OptunaTuner(
            _grid_cfg(sampler="tpe"),
            spaces,
            lambda trial, params: 0.0,
        )
        tuner.study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=7)
        )
        return tuner

    def test_no_trials_and_fitting_default_enqueues(self) -> None:
        tuner = self._tuner([_x_space(default=1)])
        assert tuner.study is not None
        tuner._enqueue_defaults()
        assert len(tuner.study.trials) == 1
        assert tuner.study.trials[0].state == optuna.trial.TrialState.WAITING
        assert tuner.study.trials[0].system_attrs["fixed_params"] == {"x": 1}

    def test_existing_trials_skip_enqueue(self) -> None:
        tuner = self._tuner([_x_space(default=1)])
        assert tuner.study is not None
        tuner.study.enqueue_trial({"x": 2})  # creates a WAITING trial
        tuner._enqueue_defaults()
        assert len(tuner.study.trials) == 1

    def test_no_declared_defaults_no_enqueue(self) -> None:
        tuner = self._tuner([_x_space(default=None)])
        assert tuner.study is not None
        tuner._enqueue_defaults()
        assert len(tuner.study.trials) == 0


# --- Pareto representative ---


def _frozen_trial(
    number: int | None, values: list[float], params: dict[str, int] | None = None
) -> optuna.trial.FrozenTrial:
    return optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=values,
        params=params or {},
        distributions={},
    )


class TestPickParetoRepresentative:
    def _mo_tuner(self, directions: list[str]) -> OptunaTuner:
        tuner = OptunaTuner(
            _grid_cfg(directions=directions), [], lambda trial, params: 0.0
        )
        tuner.study = optuna.create_study(directions=directions)
        return tuner

    def _front_with(
        self, tuner: OptunaTuner, values_list: list[list[float]]
    ) -> list[optuna.trial.FrozenTrial]:
        # add_trial assigns real trial numbers in insertion order
        assert tuner.study is not None
        tuner.study.add_trials([_frozen_trial(None, values) for values in values_list])
        return list(tuner.study.trials)

    def test_best_first_objective_wins(self) -> None:
        tuner = self._mo_tuner(["maximize", "minimize"])
        pareto = self._front_with(tuner, [[0.8, 0.1], [0.9, 0.5]])
        assert tuner._pick_pareto_representative(pareto).number == 1

    def test_tie_on_first_objective_broken_by_lowest_number(self) -> None:
        tuner = self._mo_tuner(["maximize", "minimize"])
        pareto = self._front_with(tuner, [[0.9, 0.2], [0.9, 0.3]])
        assert tuner._pick_pareto_representative(pareto).number == 0

    def test_minimize_first_objective_prefers_lowest(self) -> None:
        tuner = self._mo_tuner(["minimize", "minimize"])
        pareto = self._front_with(tuner, [[0.4, 0.1], [0.2, 0.9]])
        assert tuner._pick_pareto_representative(pareto).number == 1

    def test_best_trial_from_pareto_front(self) -> None:
        tuner = OptunaTuner(
            _grid_cfg(
                directions=["maximize", "minimize"],
                n_trials=4,
                sampler_kwargs={"search_space": {"x": [1, 2, 3, 4]}},
            ),
            [_x_space(low=1, high=4)],
            lambda trial, params: [float(params["x"]), 1.0 / params["x"]],
        )
        tuner.search()
        assert tuner.study is not None
        assert len(tuner.study.directions) == 2
        # [4, 0.25] is the front's best on the first (maximize) objective
        best = tuner._best_trial()
        assert best is not None
        assert best.values == [4.0, 0.25]
        assert tuner._best_params() == {"x": 4}


# --- _raise_if_all_failed ---


class TestRaiseIfAllFailed:
    def _tuner(self) -> OptunaTuner:
        tuner = OptunaTuner(_grid_cfg(), [], lambda trial, params: 0.0)
        tuner.study = optuna.create_study(direction="maximize")
        return tuner

    def test_all_pruned_is_not_failure(self) -> None:
        tuner = self._tuner()
        assert tuner.study is not None
        tuner.study.add_trial(
            optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED)
        )
        tuner.study.add_trial(
            optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED)
        )
        tuner._raise_if_all_failed()  # must not raise

    def test_all_failed_raises_with_traceback(self) -> None:
        tuner = self._tuner()
        assert tuner.study is not None
        tuner.study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
                user_attrs={"error": "ValueError('boom')", "traceback": "TB-TEXT"},
            )
        )
        with pytest.raises(RuntimeError, match="1 trial\\(s\\) failed"):
            tuner._raise_if_all_failed()

    def test_error_detail_includes_traceback(self) -> None:
        tuner = self._tuner()
        assert tuner.study is not None
        tuner.study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
                user_attrs={"error": "ValueError('boom')", "traceback": "TB-TEXT"},
            )
        )
        with pytest.raises(RuntimeError, match="TB-TEXT"):
            tuner._raise_if_all_failed()

    def test_any_completed_passes(self) -> None:
        tuner = self._tuner()
        assert tuner.study is not None
        tuner.study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.FAIL,
                user_attrs={"error": "e"},
            )
        )
        tuner.study.add_trial(_frozen_trial(1, [0.5]))
        tuner._raise_if_all_failed()


# --- _compute_param_importances ---


class TestComputeParamImportances:
    def test_fewer_than_four_completed_returns_none(self) -> None:
        tuner = OptunaTuner(
            _grid_cfg(), [_x_space()], lambda trial, params: float(params["x"])
        )
        tuner.search()
        assert tuner.study is not None
        assert len([t for t in tuner.study.trials if t.state.name == "COMPLETE"]) == 2
        assert tuner._param_importances is None

    def test_five_completed_returns_dict(self) -> None:
        cfg = _grid_cfg(
            sampler_kwargs={"search_space": {"x": [1, 2, 3, 4, 5]}}, n_trials=5
        )
        tuner = OptunaTuner(
            cfg, [_x_space(low=1, high=5)], lambda trial, params: float(params["x"])
        )
        tuner.search()
        assert isinstance(tuner._param_importances, dict)
        assert set(tuner._param_importances) == {"x"}
        assert tuner._param_importances["x"] == pytest.approx(1.0)


# --- _save_results ---


class TestSaveResults:
    def test_artifacts_written(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "opt"
        cfg = _grid_cfg(save_dir=str(save_dir))
        tuner = OptunaTuner(cfg, [_x_space()], lambda trial, params: float(params["x"]))
        tuner.search()

        best = yaml.safe_load((save_dir / "best_params.yaml").read_text())
        assert best == {"x": 2}

        history = yaml.safe_load((save_dir / "search_history.yaml").read_text())
        assert len(history) == 2
        assert {row["state"] for row in history} == {"COMPLETE"}
        assert sorted(row["params"]["x"] for row in history) == [1, 2]
        assert all("value" in row for row in history)

        echo = yaml.safe_load((save_dir / "optuna_config.yaml").read_text())
        assert echo["study_name"] == "utest_study"

    def test_multi_objective_history_saves_values_list(self, tmp_path: Path) -> None:
        # FrozenTrial.value RAISES on multi-objective studies (optuna >= 4);
        # the history must branch on the study shape and save the values list.
        save_dir = tmp_path / "opt_mo"
        cfg = _grid_cfg(save_dir=str(save_dir), directions=["maximize", "minimize"])
        tuner = OptunaTuner(
            cfg,
            [_x_space()],
            lambda trial, params: [float(params["x"]), 1.0 / params["x"]],
        )
        tuner.search()

        history = yaml.safe_load((save_dir / "search_history.yaml").read_text())
        assert len(history) == 2
        assert all(isinstance(row["value"], list) for row in history)
        assert all(len(row["value"]) == 2 for row in history)
        assert {row["value"][0] for row in history} == {1.0, 2.0}
