"""Tests for the RunConfig yaml archive: lossless round-trip and validation."""

import pytest

from utils.config.archive import (
    load_run_config_archive,
    load_run_metadata,
    save_run_config_archive,
)
from utils.config.run_config import config_to_dict
from utils.core import MODEL_CONFIGS


class TestSaveLoadRoundTrip:
    def test_round_trip_is_field_equal(self, make_run_config, tmp_path):
        rc = make_run_config(model_kwargs={"hidden_dim": 32, "dropout": 0.25})
        config_path, metadata_path = save_run_config_archive(rc, tmp_path)
        loaded = load_run_config_archive(config_path)
        assert config_to_dict(loaded) == config_to_dict(rc)
        assert metadata_path is None

    def test_concrete_model_class_preserved(self, make_run_config, tmp_path):
        rc = make_run_config()
        config_path, _ = save_run_config_archive(rc, tmp_path)
        loaded = load_run_config_archive(config_path)
        assert type(loaded.model) is type(rc.model)
        assert isinstance(loaded.model, MODEL_CONFIGS._registry["TinyTestModel"])

    def test_none_floats_and_lists_survive(self, make_run_config, tmp_path):
        rc = make_run_config()
        rc.general.log_dir = None
        rc.general.seed = 7
        rc.model.learning_rate = 1e-3
        config_path, _ = save_run_config_archive(rc, tmp_path)
        loaded = load_run_config_archive(config_path)
        assert loaded.general.log_dir is None
        assert loaded.general.seed == 7
        assert loaded.model.learning_rate == pytest.approx(1e-3)
        assert loaded.data.sample_attempts_bins == [20, 100]

    def test_progress_round_trips(self, make_run_config, tmp_path):
        config_path, _ = save_run_config_archive(
            make_run_config(progress="none"), tmp_path
        )
        loaded = load_run_config_archive(config_path)
        assert loaded.general.progress == "none"

    def test_save_creates_log_dir(self, make_run_config, tmp_path):
        nested = tmp_path / "runs" / "exp"
        config_path, _ = save_run_config_archive(make_run_config(), nested)
        assert config_path.exists()
        assert config_path.name == "run_config.yaml"


class TestValidation:
    def test_unknown_top_level_node_raises(self, make_run_config, tmp_path):
        import yaml

        config_path, _ = save_run_config_archive(make_run_config(), tmp_path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data["mystery_node"] = {"x": 1}
        config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(TypeError, match="mystery_node"):
            load_run_config_archive(config_path)

    def test_unknown_field_within_node_raises(self, make_run_config, tmp_path):
        import yaml

        config_path, _ = save_run_config_archive(make_run_config(), tmp_path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data["general"]["renamed_field"] = True
        config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(TypeError, match="renamed_field"):
            load_run_config_archive(config_path)

    def test_missing_nodes_filled_from_defaults(self, tmp_path):
        import yaml

        # Only the experiment node — everything else must come from defaults.
        path = tmp_path / "run_config.yaml"
        path.write_text(
            yaml.safe_dump({"experiment": {"model_name": "TinyTestModel"}}),
            encoding="utf-8",
        )
        rc = load_run_config_archive(path)
        assert rc.general.seed == 42
        assert rc.model.epochs == 2  # TinyTestModelConfig default

    def test_missing_progress_field_falls_back_to_auto(self, make_run_config, tmp_path):
        import yaml

        config_path, _ = save_run_config_archive(make_run_config(), tmp_path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        del data["general"]["progress"]  # archive written by an older version
        config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        rc = load_run_config_archive(config_path)
        assert rc.general.progress == "auto"


class TestMetadata:
    def test_sidecar_written_and_round_trips(self, make_run_config, tmp_path):
        meta = {"total_params": 1234, "optimizer": "Adam", "device": "cpu"}
        _, metadata_path = save_run_config_archive(
            make_run_config(), tmp_path, metadata=meta
        )
        assert metadata_path is not None and metadata_path.exists()
        assert load_run_metadata(tmp_path) == meta

    def test_missing_sidecar_returns_empty(self, tmp_path):
        assert load_run_metadata(tmp_path) == {}
