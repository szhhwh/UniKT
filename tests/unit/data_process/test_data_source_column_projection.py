"""Tests for split-sequence column projection: _resolve_split_columns + getters."""

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from utils.data_process.data_source import DataSource

# Mirrors a rich-column split parquet (assistments-style schema).
_ROWS = 6
_CORE = ("sequence_id", "seq_pos", "label", "user", "question", "interaction_id")


def _write_split_parquet(data_folder: Path, name: str, *, skill: bool) -> None:
    frame = {
        "interaction_id": pl.Series(range(_ROWS), dtype=pl.Int64),
        "user": pl.Series([0] * 3 + [1] * 3, dtype=pl.Int32),
        "question": pl.Series([1, 2, 3] * 2, dtype=pl.Int32),
        "label": pl.Series([1, 0, 1] * 2, dtype=pl.Int8),
        "attempt_count": pl.Series([1] * _ROWS, dtype=pl.Int64),
        "hint_count": pl.Series([0] * _ROWS, dtype=pl.Int64),
        "ms_first_response": pl.Series([1000] * _ROWS, dtype=pl.Int64),
        "timestamp": pl.Series(range(_ROWS), dtype=pl.Int64),
        "fold": pl.Series([0] * 3 + [1] * 3, dtype=pl.Int32),
        "sequence_id": pl.Series([0] * 3 + [1] * 3, dtype=pl.Int32),
        "seq_pos": pl.Series([0, 1, 2] * 2, dtype=pl.Int64),
    }
    if skill:
        frame["skill"] = pl.Series([10, 11, 12] * 2, dtype=pl.Int32)
    else:
        frame["skills"] = pl.Series([[10], [11], [12]] * 2, dtype=pl.List(pl.Int32))
    pl.DataFrame(frame).write_parquet(data_folder / f"stub_{name}.parquet")


class TestResolveSplitColumns:
    def test_empty_declaration_returns_structural_core_with_fold(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_question_sequence", skill=False)
        ds = make_data_source(data_folder=tmp_path)

        assert ds._resolve_split_columns(
            "split_question_sequence", required=(), optional=()
        ) == [*_CORE, "skills", "fold"]

    def test_skill_chain_includes_skill_column(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_skill_sequence", skill=True)
        ds = make_data_source(data_folder=tmp_path)

        columns = ds._resolve_split_columns(
            "split_skill_sequence", required=(), optional=()
        )
        assert columns == [*_CORE, "skill", "fold"]

    def test_no_fold_column_in_schema_is_omitted(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_question_sequence", skill=False)
        pl.read_parquet(tmp_path / "stub_split_question_sequence.parquet").drop(
            "fold"
        ).write_parquet(tmp_path / "stub_split_question_sequence.parquet")
        ds = make_data_source(data_folder=tmp_path)

        assert "fold" not in ds._resolve_split_columns(
            "split_question_sequence", required=(), optional=()
        )

    def test_optional_columns_intersected_with_schema(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_question_sequence", skill=False)
        ds = make_data_source(data_folder=tmp_path)

        columns = ds._resolve_split_columns(
            "split_question_sequence",
            required=(),
            optional=("timestamp", "nonexistent_col"),
        )
        assert "timestamp" in columns
        assert "nonexistent_col" not in columns

    def test_missing_required_column_raises_with_dataset_name(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_question_sequence", skill=False)
        ds = make_data_source(data_folder=tmp_path, dataset="stub")

        with pytest.raises(ValueError, match=r"stub.*nonexistent_col"):
            ds._resolve_split_columns(
                "split_question_sequence",
                required=("nonexistent_col",),
                optional=(),
            )

    def test_missing_parquet_raises_actionable_error(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        ds = make_data_source(data_folder=tmp_path)

        with pytest.raises(FileNotFoundError, match=r"data_process\.py process"):
            ds.get_split_question_sequence_data()


class TestSplitGetterProjection:
    def test_question_getter_projects_and_preserves_rows(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_question_sequence", skill=False)
        ds = make_data_source(data_folder=tmp_path)

        df = ds.get_split_question_sequence_data()
        assert df.columns == [*_CORE, "skills", "fold"]
        assert df.height == _ROWS

    def test_feature_columns_loaded_via_required(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_question_sequence", skill=False)
        ds = make_data_source(data_folder=tmp_path)

        df = ds.get_split_question_sequence_data(required=("ms_first_response",))
        assert "ms_first_response" in df.columns
        assert "attempt_count" not in df.columns

    def test_distinct_column_sets_cached_separately(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        _write_split_parquet(tmp_path, "split_question_sequence", skill=False)
        ds = make_data_source(data_folder=tmp_path)

        core = ds.get_split_question_sequence_data()
        rich = ds.get_split_question_sequence_data(required=("timestamp",))
        again = ds.get_split_question_sequence_data()

        # Same projection hits the cache entry; distinct projections coexist.
        assert len(ds._data_cache) == 2
        assert again is core
        assert rich is not core
