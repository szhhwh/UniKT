"""Tests for compute_user_metrics: formula checks on synthetic users."""

import math

import pandas as pd

from utils.case_analysis.user_metrics import compute_user_metrics


def _df(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "label", "prediction"], dtype=float)


def test_all_correct_single_class_user() -> None:
    df = _df([(1, 1.0, 1.0), (1, 1.0, 1.0), (1, 1.0, 1.0)])
    m = compute_user_metrics(df).iloc[0]
    assert m["num_attempts"] == 3
    assert m["correct_rate"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["error_rate"] == 0.0
    assert math.isnan(m["auc"])  # single-class label: AUC undefined


def test_mixed_user_formulas() -> None:
    df = _df(
        [
            (2, 1.0, 0.9),  # correct, confident
            (2, 0.0, 0.2),  # correct rejection
            (2, 1.0, 0.4),  # wrong
            (2, 0.0, 0.6),  # wrong
        ]
    )
    m = compute_user_metrics(df).iloc[0]
    assert m["num_attempts"] == 4
    assert m["correct_rate"] == 0.5
    assert m["predicted_correct_rate"] == (0.9 + 0.2 + 0.4 + 0.6) / 4
    assert (
        m["accuracy"] == 0.5
    )  # round(0.9)=1, round(0.2)=0, round(0.4)=0, round(0.6)=1
    assert m["avg_confidence"] == (0.9 + 0.8 + 0.4 + 0.4) / 4
    assert m["calibration_error"] == abs(m["predicted_correct_rate"] - 0.5)
    assert 0.0 < m["auc"] < 1.0


def test_per_user_rows() -> None:
    df = _df([(1, 1.0, 1.0), (2, 0.0, 0.0), (2, 1.0, 1.0)])
    m = compute_user_metrics(df)
    assert list(m["user_id"]) == [1, 2]
    assert list(m["num_attempts"]) == [1, 2]
