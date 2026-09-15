"""Tests for ConfigParser: short flags, model resolution precedence, typed output."""

from dataclasses import dataclass

import pytest

from utils.config.config_parser import (
    ConfigParser,
    _read_model_name,
    build_node,
    expand_short_flags,
    parse_run_archive,
    peek_flag_value,
)
from utils.config.run_config import RunConfig
from utils.core import MODEL_CONFIGS
from utils.core.registry import register_model_config


def _parse(argv, **parser_kwargs):
    return ConfigParser(**parser_kwargs).parse_args(argv)


# --- short flag expansion ---


class TestExpandShortFlags:
    def test_dash_m_with_value(self):
        assert expand_short_flags(["-m", "DKT"]) == [
            "--experiment.model_name",
            "DKT",
        ]

    def test_dash_m_inline_equals(self):
        assert expand_short_flags(["-m=DKT"]) == ["--experiment.model_name=DKT"]

    def test_dash_d_forms(self):
        assert expand_short_flags(["-d", "assist09"]) == [
            "--data.dataset",
            "assist09",
        ]
        assert expand_short_flags(["-d=assist09"]) == ["--data.dataset=assist09"]

    def test_combined_flags(self):
        out = expand_short_flags(["-m", "DKT", "-d", "assist09", "--general.seed", "1"])
        assert out == [
            "--experiment.model_name",
            "DKT",
            "--data.dataset",
            "assist09",
            "--general.seed",
            "1",
        ]

    def test_trailing_bare_dash_m_kept(self):
        # No value follows: left as-is so argparse surfaces the usage error.
        assert expand_short_flags(["-d", "x", "-m"]) == [
            "--data.dataset",
            "x",
            "-m",
        ]

    def test_unknown_tokens_untouched(self):
        argv = ["--config", "c.yaml", "positional"]
        assert expand_short_flags(argv) == argv


# --- model resolution ---


class TestModelResolution:
    def test_explicit_m_beats_config_yaml(
        self, tiny_model_config_name, write_model_yaml, registry_snapshot
    ):
        @register_model_config("AltTestModel")
        class AltTestModel(MODEL_CONFIGS._registry[tiny_model_config_name]): ...

        yaml_path = write_model_yaml(model_name="AltTestModel")
        rc = _parse(["-m", "TinyTestModel", "-d", "tinyds", "--config", str(yaml_path)])
        assert rc.experiment.model_name == "TinyTestModel"
        assert rc.model.hidden_dim == 8  # TinyTestModel's field, not Alt's

    def test_config_yaml_beats_default_config(self, write_model_yaml):
        default = write_model_yaml(model_name="TinyTestModel", filename="default.yaml")
        override = write_model_yaml(
            model_name="TinyTestModel",
            overrides={"model.hidden_dim": 64},
            filename="override.yaml",
        )
        rc = _parse(
            ["-d", "tinyds", "--config", str(override)],
            default_config=str(default),
        )
        assert rc.model.hidden_dim == 64

    def test_default_config_used_when_nothing_else(
        self, tiny_model_config_name, write_model_yaml
    ):
        default = write_model_yaml(
            overrides={"model.dropout": 0.5}, filename="default.yaml"
        )
        rc = _parse(["-d", "tinyds"], default_config=str(default))
        assert rc.experiment.model_name == "TinyTestModel"
        assert rc.model.dropout == pytest.approx(0.5)

    def test_missing_model_exits_with_message(self):
        with pytest.raises(SystemExit) as exc:
            _parse(["-d", "tinyds"])
        # SystemExit carries the full message, including the available-models list.
        assert "model name is required" in str(exc.value.code)
        assert "available:" in str(exc.value.code)

    def test_help_without_model_prints_framework_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _parse(["-h"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--general.seed" in out
        assert "--data.dataset" in out


# --- parsing ---


class TestParsing:
    def test_flags_land_on_typed_nodes(self, tiny_model_config_name):
        rc = _parse(
            [
                "-m",
                "TinyTestModel",
                "-d",
                "tinyds",
                "--general.skip_test",
                "true",
                "--general.seed",
                "7",
                "--model.hidden_dim",
                "16",
            ]
        )
        assert isinstance(rc, RunConfig)
        assert rc.general.skip_test is True
        assert rc.general.seed == 7
        assert rc.model.hidden_dim == 16

    def test_literal_choice_rejects_invalid_value(self, tiny_model_config_name):
        with pytest.raises(SystemExit):
            _parse(
                [
                    "-m",
                    "TinyTestModel",
                    "-d",
                    "tinyds",
                    "--data.sample_strategy",
                    "bogus",
                ]
            )

    def test_literal_choice_accepts_valid_value(self, tiny_model_config_name):
        rc = _parse(
            ["-m", "TinyTestModel", "-d", "tinyds", "--data.sample_strategy", "time"]
        )
        assert rc.data.sample_strategy == "time"

    def test_progress_defaults_to_auto(self, tiny_model_config_name):
        rc = _parse(["-m", "TinyTestModel", "-d", "tinyds"])
        assert rc.general.progress == "auto"

    def test_progress_flag_accepts_valid_value(self, tiny_model_config_name):
        rc = _parse(
            ["-m", "TinyTestModel", "-d", "tinyds", "--general.progress", "none"]
        )
        assert rc.general.progress == "none"

    def test_progress_flag_rejects_invalid_value(self, tiny_model_config_name):
        with pytest.raises(SystemExit):
            _parse(
                ["-m", "TinyTestModel", "-d", "tinyds", "--general.progress", "bogus"]
            )

    def test_missing_dataset_exits(self, tiny_model_config_name):
        with pytest.raises(SystemExit) as exc:
            _parse(["-m", "TinyTestModel"])
        assert "dataset is required" in str(exc.value.code)

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError):
            _parse(["-m", "NoSuchModel", "-d", "tinyds"])


# --- require_dataset (direct; data_process.py calls it on its own parser) ---


class TestRequireDataset:
    def test_empty_dataset_exits_with_choices(self):
        from dataclasses import dataclass

        from utils.config import require_dataset

        @dataclass
        class _Data:
            dataset: str = ""

        class _RC:
            data = _Data()

        with pytest.raises(SystemExit) as exc:
            require_dataset(_RC())
        assert "dataset is required" in str(exc.value.code)
        assert "-d/--data.dataset" in str(exc.value.code)

    def test_nonempty_dataset_passes(self, make_run_config):
        from utils.config import require_dataset

        require_dataset(make_run_config())  # dataset="tinyds" — no raise


# --- _read_model_name ---


class TestReadModelName:
    def test_valid_yaml(self, write_model_yaml):
        assert _read_model_name(str(write_model_yaml())) == "TinyTestModel"

    def test_missing_file_returns_none(self, tmp_path):
        assert _read_model_name(str(tmp_path / "nope.yaml")) is None

    def test_malformed_yaml_returns_none(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("experiment: [unclosed", encoding="utf-8")
        assert _read_model_name(str(bad)) is None

    def test_yaml_without_model_name_returns_none(self, tmp_path):
        plain = tmp_path / "plain.yaml"
        plain.write_text("general:\n  seed: 1\n", encoding="utf-8")
        assert _read_model_name(str(plain)) is None


# --- peek_flag_value ---


class TestPeekFlagValue:
    def test_space_form(self):
        assert peek_flag_value(["--x.run_dir", "/a"], "--x.run_dir") == "/a"

    def test_equals_form(self):
        assert peek_flag_value(["--x.run_dir=/a"], "--x.run_dir") == "/a"

    def test_last_occurrence_wins(self):
        argv = ["--x.run_dir", "/first", "--x.run_dir=/second"]
        assert peek_flag_value(argv, "--x.run_dir") == "/second"

    def test_missing_flag_returns_none(self):
        assert peek_flag_value(["--other", "v"], "--x.run_dir") is None

    def test_trailing_bare_flag_has_no_value(self):
        assert peek_flag_value(["--x.run_dir"], "--x.run_dir") is None


# --- parse_run_archive ---


@dataclass
class _EntryConfig:
    run_dir: str
    checkpoint: str = "best_model.pth"


class TestParseRunArchive:
    @staticmethod
    def _parse(argv):
        return parse_run_archive(
            argv,
            prog="evaluate.py",
            description="test entry point",
            entry_node="evaluate",
            entry_cls=_EntryConfig,
        )

    def test_restores_archive_and_applies_overrides(self, make_run_archive):
        run_dir = make_run_archive(overrides={"model.batch_size": 64})
        rc, entry, resolved = self._parse(
            ["--evaluate.run_dir", str(run_dir), "--model.batch_size", "32"]
        )
        assert rc.experiment.model_name == "TinyTestModel"  # from the archive
        assert rc.data.dataset == "tinyds"
        assert rc.model.batch_size == 32  # CLI flag beats the archived value
        assert entry.checkpoint == "best_model.pth"  # entry-node default
        assert entry.run_dir == str(run_dir)
        assert resolved == run_dir.resolve()

    def test_checkpoint_entry_flag(self, make_run_archive):
        run_dir = make_run_archive()
        _, entry, _ = self._parse(
            ["--evaluate.run_dir", str(run_dir), "--evaluate.checkpoint", "last.pth"]
        )
        assert entry.checkpoint == "last.pth"

    def test_missing_run_dir_exits_with_message(self, tiny_model_config_name):
        with pytest.raises(SystemExit) as exc:
            self._parse(["--general.seed", "1"])
        assert "--evaluate.run_dir is required" in str(exc.value.code)

    def test_help_without_run_dir_prints_framework_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            self._parse(["-h"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "--evaluate.run_dir" in out  # entry node joins the reference
        assert "--general.seed" in out

    def test_model_flag_rejected(self, make_run_archive):
        run_dir = make_run_archive()
        with pytest.raises(SystemExit) as exc:
            self._parse(["-m", "OtherModel", "--evaluate.run_dir", str(run_dir)])
        assert "cannot be combined" in str(exc.value.code)

    def test_missing_archive_exits(self, tmp_path):
        empty = tmp_path / "empty_run"
        empty.mkdir()
        with pytest.raises(SystemExit) as exc:
            self._parse(["--evaluate.run_dir", str(empty)])
        assert "run_config.yaml not found" in str(exc.value.code)


# --- extras & build_node ---


class TestExtras:
    def test_extra_node_stays_in_namespace(self, tiny_model_config_name):
        from dataclasses import dataclass

        @dataclass
        class ExtraNode:
            flag: bool = False

        parser = ConfigParser(extra_nodes={"extra": ExtraNode})
        rc, ns = parser.parse_with_extras(
            ["-m", "TinyTestModel", "-d", "tinyds", "--extra.flag", "true"]
        )
        assert ns["extra"]["flag"] is True
        assert not hasattr(rc, "extra")  # not part of RunConfig

    def test_build_node_recurses_into_dataclass_fields(self):
        from dataclasses import dataclass

        from jsonargparse import Namespace

        @dataclass
        class Inner:
            x: int = 0

        @dataclass
        class Outer:
            general: Inner = None
            scalar: str = "s"

        node = build_node(Outer, Namespace(general=Namespace(x=5), scalar="v"))
        assert isinstance(node.general, Inner)
        assert node.general.x == 5
        assert node.scalar == "v"

    def test_build_node_scalar_path(self):
        from dataclasses import dataclass

        from jsonargparse import Namespace

        @dataclass
        class FlatNode:
            seed: int = 0
            name: str = ""

        node = build_node(FlatNode, Namespace(seed=3, name="x"))
        assert node.seed == 3
        assert node.name == "x"
