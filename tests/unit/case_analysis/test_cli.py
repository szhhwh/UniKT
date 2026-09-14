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
        "min_confidence": None,
        "max_confidence": None,
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


def test_filter_supported_options_drops_unsupported():
    class _Sel:
        def select(self, results, *, min_seq_len=20, max_users=20):
            return []

    opts = cli._filter_supported_options(
        _Sel, {"min_seq_len": 5, "error_rate_range": (0.1, 0.9), "max_users": 3}
    )
    assert opts == {"min_seq_len": 5, "max_users": 3}


# --- _select_options: sentinel-None deferral to the plugin signature ---


class _SignatureSel:
    """Selector stand-in mirroring the built-in plugins' select signature."""

    def select(
        self,
        results,
        *,
        min_seq_len: int = 20,
        error_rate_range: tuple = (0.1, 0.9),
        confidence_range: tuple = (0.3, 0.95),
        max_users: int = 20,
    ):
        return []


def test_select_all_none_defers_to_plugin_defaults(run_dir):
    case = _select_args(
        run_dir,
        "diverse",
        num_users=None,
        min_seq_len=None,
        min_error=None,
        max_error=None,
    )
    assert cli._select_options(case, _SignatureSel) == {}


def test_select_partial_range_merges_plugin_default(run_dir):
    case = _select_args(run_dir, "diverse", min_error=0.2, max_error=None)
    opts = cli._select_options(case, _SignatureSel)
    assert opts["error_rate_range"] == (0.2, 0.9)  # 0.9 from the signature


def test_select_confidence_window_passed(run_dir):
    case = _select_args(run_dir, "diverse", min_confidence=0.4, max_confidence=0.9)
    opts = cli._select_options(case, _SignatureSel)
    assert opts["confidence_range"] == (0.4, 0.9)


def test_cmd_select_defaults_defer_to_plugin(run_dir):
    cli.cmd_select(
        _select_args(
            run_dir,
            "diverse",
            num_users=None,
            min_seq_len=None,
            min_error=None,
            max_error=None,
        )
    )
    path = run_dir / "case_analysis" / "diverse" / "selected_users.json"
    records = json.loads(path.read_text())
    # Plugin default max_users=20, not the old hand-written CLI default of 10.
    assert 10 < len(records) <= 20


# --- dispatch ---


def test_main_without_subcommand_prints_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["case_analysis.py"])
    cli.main()
    out = capsys.readouterr().out
    assert "inference" in out
    assert "select" in out
    assert "plot" in out
