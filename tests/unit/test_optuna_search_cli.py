"""Tests for optuna_search.py's CLI layer: reflective entry node + RunConfig.

Only ``_parse()`` is exercised — ``main()`` needs the full training stack.
"""

import pytest

from optuna_search import OptunaSearchConfig, _parse


class TestOptunaSearchParse:
    def test_defaults(self, tiny_model_config_name):
        rc, opt, metrics = _parse(["-m", tiny_model_config_name, "-d", "tinyds"])
        assert rc.experiment.model_name == tiny_model_config_name
        assert opt == OptunaSearchConfig()
        assert metrics == ["auc"]

    def test_entry_flags_roundtrip(self, tiny_model_config_name):
        _, opt, metrics = _parse(
            [
                "-m",
                tiny_model_config_name,
                "-d",
                "tinyds",
                "--optuna_search.optuna_config",
                "cfg.yaml",
                "--optuna_search.metric",
                "auc,rmse",
                "--optuna_search.resume",
                "runs/old",
                "--optuna_search.keep_trial_artifacts",
                "true",
                "--optuna_search.output_dir",
                "out",
            ]
        )
        assert opt.optuna_config == "cfg.yaml"
        assert opt.resume == "runs/old"
        assert opt.keep_trial_artifacts is True
        assert opt.output_dir == "out"
        assert metrics == ["auc", "rmse"]

    def test_runconfig_flags_share_the_parser(self, tiny_model_config_name):
        rc, _, _ = _parse(
            ["-m", tiny_model_config_name, "-d", "tinyds", "--model.batch_size", "32"]
        )
        assert rc.model.batch_size == 32

    def test_invalid_metric_exits_with_choices(self, tiny_model_config_name):
        with pytest.raises(SystemExit, match="invalid metric"):
            _parse(
                [
                    "-m",
                    tiny_model_config_name,
                    "-d",
                    "tinyds",
                    "--optuna_search.metric",
                    "bogus",
                ]
            )
