"""Tests for DataFrameSink: key contract, passthrough, parquet roundtrip."""

from pathlib import Path

import pytest

from utils.case_analysis.sinks.dataframe_sink import (
    DataFrameSink,
    get_user_sequence,
    load_case_results,
)

_FULL_BATCH = {
    "user_ids": [1, 1, 2],
    "question_ids": [10, 11, 10],
    "labels": [1, 0, 1],
    "predictions": [1, 0, 1],
    "logits": [0.9, -0.3, 0.5],
    "skills": [[0, 1], [1], [2]],
    "mask": [1, 1, 1],
    "knowledge_states": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    "custom_extra": ["a", "b", "c"],
}


def test_result_columns_position_and_passthrough() -> None:
    sink = DataFrameSink()
    sink.add_batch(_FULL_BATCH)
    sink.add_batch(
        {
            "user_ids": [2],
            "question_ids": [11],
            "labels": [0],
            "predictions": [0],
            "logits": [-0.7],
            "skills": [[0]],
            "mask": [1],
            "knowledge_states": [[0.7, 0.8]],
            "custom_extra": ["d"],
        }
    )
    df = sink.result()

    assert list(df["position"]) == [0, 1, 0, 1]
    assert list(df["custom_extra"]) == ["a", "b", "c", "d"]
    assert set(df.columns) == {
        "user_id",
        "question_id",
        "label",
        "prediction",
        "logit",
        "skill",
        "mask",
        "knowledge_state",
        "custom_extra",
        "position",
    }


def test_missing_required_key_named_in_error() -> None:
    sink = DataFrameSink()
    with pytest.raises(ValueError, match="question_ids"):
        sink.add_batch({"user_ids": [1], "labels": [0], "predictions": [0]})


def test_inconsistent_batch_lengths_rejected() -> None:
    sink = DataFrameSink()
    with pytest.raises(ValueError, match="lengths"):
        sink.add_batch(
            {
                "user_ids": [1],
                "question_ids": [10, 11],
                "labels": [0],
                "predictions": [0],
            }
        )


def test_inconsistent_keys_across_batches_rejected() -> None:
    sink = DataFrameSink()
    sink.add_batch(
        {"user_ids": [1], "question_ids": [10], "labels": [0], "predictions": [0]}
    )
    with pytest.raises(ValueError, match="across batches"):
        sink.add_batch(
            {
                "user_ids": [2],
                "question_ids": [11],
                "labels": [1],
                "predictions": [1],
                "skills": [[3]],
            }
        )


def test_optional_keys_absent_leaves_no_column() -> None:
    sink = DataFrameSink()
    sink.add_batch(
        {"user_ids": [5], "question_ids": [3], "labels": [1], "predictions": [1]}
    )
    df = sink.result()
    assert "logit" not in df.columns
    assert "knowledge_state" not in df.columns
    assert "mask" not in df.columns


def test_parquet_roundtrip_nested_lists(tmp_path: Path) -> None:
    sink = DataFrameSink()
    sink.add_batch(_FULL_BATCH)
    p = tmp_path / "predictions.parquet"
    DataFrameSink.save(sink.result(), str(p))

    df = load_case_results(str(p))
    assert [list(x) for x in df["skill"]] == [[0, 1], [1], [2]]
    assert [list(x) for x in df["knowledge_state"]] == [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    assert list(get_user_sequence(df, 1)["question_id"]) == [10, 11]


def test_parquet_roundtrip_without_knowledge_state(tmp_path: Path) -> None:
    sink = DataFrameSink()
    sink.add_batch(
        {
            "user_ids": [5, 5],
            "question_ids": [3, 4],
            "labels": [1, 0],
            "predictions": [1, 0],
        }
    )
    p = tmp_path / "nk.parquet"
    DataFrameSink.save(sink.result(), str(p))
    df = load_case_results(str(p))
    assert list(df["position"]) == [0, 1]


def test_load_rejects_missing_canonical_columns(tmp_path: Path) -> None:
    import pandas as pd

    p = tmp_path / "bad.parquet"
    pd.DataFrame({"foo": [1]}).to_parquet(p)
    with pytest.raises(ValueError, match="user_id"):
        load_case_results(str(p))
