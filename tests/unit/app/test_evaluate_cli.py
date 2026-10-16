"""Tests for evaluate.py's CLI layer: archive restoration + reflective overrides.

Only ``_parse()`` is exercised — ``main()`` needs the full training stack. The
shared ``make_run_archive`` fixture supplies a minimal ``run_config.yaml``.
"""

from collections.abc import Callable
from pathlib import Path

from evaluate import EvaluateConfig, _parse


class TestEvaluateParse:
    def test_restores_archive_values(
        self, make_run_archive: Callable[..., Path]
    ) -> None:
        run_dir = make_run_archive(
            overrides={"model.batch_size": 64, "general.seed": 123}
        )
        rc, ev_cfg, resolved = _parse(["--evaluate.run_dir", str(run_dir)])
        assert rc.experiment.model_name == "TinyTestModel"
        assert rc.model.batch_size == 64
        assert rc.general.seed == 123
        assert ev_cfg == EvaluateConfig(run_dir=str(run_dir))
        assert resolved == run_dir.resolve()

    def test_reflective_flags_override_archive(
        self, make_run_archive: Callable[..., Path]
    ) -> None:
        run_dir = make_run_archive(overrides={"model.batch_size": 64})
        rc, _, _ = _parse(
            [
                "--evaluate.run_dir",
                str(run_dir),
                "--general.device",
                "cpu",
                "--model.batch_size",
                "32",
                "--data.data_base_path",
                "/somewhere",
            ]
        )
        assert rc.general.device == "cpu"
        assert rc.model.batch_size == 32
        assert rc.data.data_base_path == "/somewhere"

    def test_checkpoint_entry_flag(self, make_run_archive: Callable[..., Path]) -> None:
        run_dir = make_run_archive()
        _, ev_cfg, _ = _parse(
            [
                "--evaluate.run_dir",
                str(run_dir),
                "--evaluate.checkpoint",
                "last.pth",
            ]
        )
        assert ev_cfg.checkpoint == "last.pth"
