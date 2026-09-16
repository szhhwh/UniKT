"""Tests for the RunConfig dataclass tree and schema assembly."""

from collections.abc import Callable

import pytest

from utils.config.run_config import (
    _FRAMEWORK_NODES,
    ModelConfig,
    RunConfig,
    RunDataConfig,
    build_run_config_schema,
    config_to_dict,
)
from utils.core import MODEL_CONFIGS


class TestDefaults:
    def test_model_config_contract_fields(self) -> None:
        cfg = ModelConfig()
        assert cfg.epochs == 150
        assert cfg.batch_size == 32
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.weight_decay == 0.0

    def test_run_config_nodes_are_independent_instances(self) -> None:
        a, b = RunConfig(), RunConfig()
        assert a.general is not b.general
        assert a.model is not b.model

    def test_list_default_fields_not_shared(self) -> None:
        a, b = RunDataConfig(), RunDataConfig()
        assert a.sample_attempts_bins == [20, 100]
        assert a.sample_correct_bins == [0.4, 0.8]
        a.sample_attempts_bins.append(999)
        assert b.sample_attempts_bins == [20, 100]  # no aliasing


class TestSchema:
    def test_framework_nodes_are_the_fixed_five(self) -> None:
        assert set(_FRAMEWORK_NODES) == {
            "general",
            "compile",
            "early_stopping",
            "experiment",
            "data",
        }

    def test_schema_binds_registered_model(self, tiny_model_config_name: str) -> None:
        schema = build_run_config_schema(tiny_model_config_name)
        assert set(schema) == set(_FRAMEWORK_NODES) | {"model"}
        assert schema["model"] is MODEL_CONFIGS._registry[tiny_model_config_name]

    def test_unknown_model_raises_registry_keyerror(self) -> None:
        # The registry's KeyError lists the available names.
        with pytest.raises(KeyError, match="not found"):
            build_run_config_schema("NoSuchModelAnywhere")


class TestConfigToDict:
    def test_none_returns_empty_dict(self) -> None:
        assert config_to_dict(None) == {}

    def test_dataclass_expands_recursively(
        self, make_run_config: Callable[..., RunConfig]
    ) -> None:
        rc = make_run_config(model_kwargs={"hidden_dim": 16})
        d = config_to_dict(rc)
        assert set(d) == {
            "general",
            "compile",
            "early_stopping",
            "experiment",
            "data",
            "model",
        }
        assert d["model"]["hidden_dim"] == 16
        assert d["general"]["seed"] == 42
        assert d["data"]["sample_attempts_bins"] == [20, 100]

    def test_mapping_passes_through_dict(self) -> None:
        assert config_to_dict({"a": 1}) == {"a": 1}
