"""Function-level tests for the case_analysis.py CLI commands."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import case_analysis as cli
from utils.case_analysis import DataFrameSink


class _FakeRegistry:
    def __init__(self, mapping):
        self._mapping = mapping

    def __contains__(self, name):
        return name in self._mapping

    def get(self, name):
        return self._mapping.get(name)

    def keys(self):
        return self._mapping.keys()


@pytest.fixture
def run_dir(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for uid in range(40):
        n = 30
        label = rng.integers(0, 2, n)
        p = float(rng.uniform(0.2, 0.9))
        pred = (rng.random(n) < p).astype(int)
        for t in range(n):
            rows.append(
                {
                    "user_id": uid,
                    "question_id": int(rng.integers(0, 50)),
                    "skill": [int(rng.integers(0, 5))],
                    "label": int(label[t]),
                    "prediction": float(pred[t]),
                    "logit": float(pred[t]) - 0.5,
                    "mask": 1,
                    "knowledge_state": [
                        round(float(rng.random()), 3) for _ in range(5)
                    ],
                }
            )
    df = pd.DataFrame(rows)
    df["position"] = df.groupby("user_id").cumcount()
    out = tmp_path / "run"
    (out / "case_analysis").mkdir(parents=True)
    DataFrameSink.save(df, str(out / "case_analysis" / "predictions.parquet"))
    return out


def test_cmd_inference_end_to_end(
    tmp_path, monkeypatch, rc, dummy_analyzer_cls, checkpoint_path
):
    import torch

    run = tmp_path / "fakemodel_run"
    run.mkdir()
    torch.save(
        dummy_analyzer_cls(rc, None, checkpoint_path).model.state_dict(),
        run / "best_model.pth",
    )

    monkeypatch.setattr(cli, "get_data_source", lambda *_: None)
    monkeypatch.setattr(
        cli, "ANALYZERS", _FakeRegistry({"DummyModel": dummy_analyzer_cls})
    )

    args = type(
        "_Args",
        (),
        {"run_dir": str(run), "checkpoint": "best_model.pth", "sink": "dataframe"},
    )()
    cli.cmd_inference(rc, args)

    assert (run / "case_analysis" / "predictions.parquet").exists()
    assert (run / "case_analysis" / "user_summaries.parquet").exists()


def test_cmd_inference_unregistered_model_exits(tmp_path, monkeypatch, rc):
    (tmp_path / "best_model.pth").write_bytes(b"x")
    monkeypatch.setattr(cli, "ANALYZERS", _FakeRegistry({}))

    args = type(
        "_Args",
        (),
        {"run_dir": str(tmp_path), "checkpoint": "best_model.pth", "sink": "dataframe"},
    )()
    with pytest.raises(SystemExit, match="no registered case analyzer"):
        cli.cmd_inference(rc, args)


def _select_args(run_dir, selector, **overrides):
    fields = {
        "run_dir": str(run_dir),
        "selector": selector,
        "num_users": 3,
        "min_seq_len": 5,
        "min_error": 0.0,
        "max_error": 1.0,
        "min_confidence": 0.0,
        "max_confidence": 1.0,
    }
    fields.update(overrides)
    return type("_Args", (), fields)()


def test_cmd_select_writes_selected_users(run_dir):
    cli.cmd_select(_select_args(run_dir, "extreme"))
    path = run_dir / "case_analysis" / "extreme" / "selected_users.json"
    assert path.exists()
    records = json.loads(path.read_text())
    assert len(records) == 3
    assert "user_id" in records[0]


def test_cmd_select_unknown_selector_exits(run_dir):
    with pytest.raises(SystemExit, match="nope"):
        cli.cmd_select(_select_args(run_dir, "nope"))


def test_cmd_plot_renders_figures(run_dir):
    cli.cmd_select(_select_args(run_dir, "extreme", num_users=2))
    cli.cmd_plot(
        type(
            "_Args",
            (),
            {
                "run_dir": str(run_dir),
                "selected_users": "extreme",
                "visualizer": "heatmap",
                "max_seq_len": 20,
            },
        )()
    )
    figs = run_dir / "case_analysis" / "extreme" / "figures"
    pngs = [f for f in figs.iterdir() if f.suffix == ".png"]
    assert len(pngs) == 2


# --- select kwargs mapping ---


def test_select_kwargs_follow_cli_config(run_dir, monkeypatch):
    captured = {}

    class _CapturingSelector:
        def select(self, results, **options):
            captured.update(options)
            return [0, 1, 2]

    monkeypatch.setattr(
        cli, "CASE_SELECTORS", _FakeRegistry({"cap": _CapturingSelector})
    )

    cli.cmd_select(_select_args(run_dir, "cap"))
    assert captured == {
        "min_seq_len": 5,
        "error_rate_range": (0.0, 1.0),
        "confidence_range": (0.0, 1.0),
        "max_users": 3,
    }


def test_cmd_select_cli_defaults_match_builtin_selectors(run_dir):
    cli.cmd_select(cli.CaseSelectConfig(run_dir=str(run_dir)))
    path = run_dir / "case_analysis" / "diverse" / "selected_users.json"
    records = json.loads(path.read_text())
    # CLI default num_users=20
    assert 10 < len(records) <= 20


# --- dispatch ---


def test_main_help_lists_subcommands(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["case_analysis.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: case_analysis.py" in out
    for cmd in ("inference", "select", "plot"):
        assert cmd in out


def test_main_without_subcommand_errors(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["case_analysis.py"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err


def test_main_unknown_subcommand_errors(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["case_analysis.py", "nope"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err
