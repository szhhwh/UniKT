"""Tests for metric-based user selectors: strategies and edge cases."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import utils.case_analysis  # noqa: F401  (registers default plugins)
from utils.case_analysis.user_metrics import compute_user_metrics
from utils.core import CASE_SELECTORS


@pytest.fixture
def results_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(60):
        n = 30
        label = rng.integers(0, 2, n)
        p = float(rng.uniform(0.2, 0.9))
        pred = (rng.random(n) < p).astype(int)
        rows.append(pd.DataFrame({"user_id": uid, "label": label, "prediction": pred}))
    return pd.concat(rows, ignore_index=True)


def _opts(**overrides: Any) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "min_seq_len": 5,
        "error_rate_range": (0.0, 1.0),
        "max_users": 10,
    }
    opts.update(overrides)
    return opts


def test_three_strategies_registered() -> None:
    assert set(CASE_SELECTORS.keys()) == {"diverse", "extreme", "random"}


def test_diverse_reproducible(results_df: pd.DataFrame) -> None:
    r1 = CASE_SELECTORS.get("diverse")().select(results_df, **_opts())
    r2 = CASE_SELECTORS.get("diverse")().select(results_df, **_opts())
    assert r1 == r2
    assert len(r1) == 10


def test_extreme_returns_top_error_rates(results_df: pd.DataFrame) -> None:
    sel = CASE_SELECTORS.get("extreme")().select(results_df, **_opts(max_users=5))
    errors = compute_user_metrics(results_df).set_index("user_id")["error_rate"]
    assert len(sel) == 5
    assert all(
        errors[u] >= errors[v] for u in sel for v in errors.index if v not in sel
    )


def test_random_respects_quota(results_df: pd.DataFrame) -> None:
    sel = CASE_SELECTORS.get("random")().select(results_df, **_opts(max_users=7))
    assert len(sel) == 7
    assert len(set(sel)) == 7


def test_filter_min_seq_len(results_df: pd.DataFrame) -> None:
    few_users = results_df[results_df["user_id"] < 3]
    # every user has exactly 30 attempts: bar above that empties the pool
    assert (
        CASE_SELECTORS.get("random")().select(few_users, **_opts(min_seq_len=31)) == []
    )
    assert (
        len(CASE_SELECTORS.get("random")().select(few_users, **_opts(min_seq_len=30)))
        == 3
    )


def test_empty_pool_returns_empty_list(results_df: pd.DataFrame) -> None:
    assert CASE_SELECTORS.get("extreme")().select(results_df, min_seq_len=1000) == []


def test_quota_above_pool_returns_all(results_df: pd.DataFrame) -> None:
    sel = CASE_SELECTORS.get("extreme")().select(results_df, **_opts(max_users=1000))
    assert 0 < len(sel) < 1000
