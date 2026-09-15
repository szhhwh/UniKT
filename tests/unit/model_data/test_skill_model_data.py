"""Tests for SkillModelData / WindowlateIterableDataset user_id propagation.

Covers the 2026-08 framework change: the windowlate iterable dataset appends
the ORIGINAL student id as a 7th tuple element, and ``build_sequence_data``
fills ``user_id_sequence`` from the split data's ``user`` column so
train/val/test share one id space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from utils.data_process import DataSource
from utils.model_data.skill_model_data import (
    SkillModelData,
    WindowlateIterableDataset,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from utils.config.run_config import RunConfig

# --- WindowlateIterableDataset ------------------------------------------------


class TestWindowlateIterableDatasetUserId:
    """7-tuple output contract: user_id last, real id on window positions."""

    @pytest.fixture
    def dataset(
        self, write_windowlate_parquet: Callable[..., str]
    ) -> WindowlateIterableDataset:
        # One sample, window length 3: positions 0..2, target at 2.
        # user_id 42 differs from any buffer/pad default to catch misalignment.
        path = write_windowlate_parquet(
            "wl.parquet",
            {
                "sample_id": [0, 0, 0],
                "position": [0, 1, 2],
                "skill": [10, 11, 12],
                "question": [5, 6, 7],
                "response": [1, 0, 0],
                "mask": [0, 0, 1],
                "user_id": [42, 42, 42],
                "group_id": [7, 7, 7],
                "true_label": [1, 0, 1],
            },
        )
        return WindowlateIterableDataset(path, max_seq_len=4)

    def test_tuple_shape_and_dtypes(self, dataset: WindowlateIterableDataset) -> None:
        sample = next(iter(dataset))
        assert len(sample) == 7
        seq, resp, mask, gid, label, q, uid = sample
        assert seq.dtype == torch.int64 and seq.shape == (4,)
        assert resp.dtype == torch.int64
        assert mask.dtype == torch.bool
        assert gid.dtype == torch.int64
        assert label.dtype == torch.int64
        assert q.dtype == torch.int64
        assert uid.dtype == torch.int64 and uid.shape == (4,)

    def test_user_id_scattered_on_window_positions(
        self, dataset: WindowlateIterableDataset
    ) -> None:
        *_, user_id = next(iter(dataset))
        # Window positions 0..2 carry the real id; tail pad position 3 keeps 0.
        assert user_id.tolist() == [42, 42, 42, 0]

    def test_user_id_not_present_in_first_six(
        self, dataset: WindowlateIterableDataset
    ) -> None:
        # Guards against reordering: first six slots keep their original
        # semantics (skill / response / target-mask / group / label / question).
        seq, resp, mask, gid, label, q, _ = next(iter(dataset))
        assert seq.tolist() == [10, 11, 12, 0]
        assert resp.tolist() == [1, 0, 0, 0]
        assert mask.tolist() == [False, False, True, False]
        assert gid.tolist() == [7, 7, 7, -1]
        assert label.tolist() == [1, 0, 1, 0]
        assert q.tolist() == [5, 6, 7, 0]

    def test_multiple_samples_user_id_constant_within_window(
        self, write_windowlate_parquet: Callable[..., str]
    ) -> None:
        path = write_windowlate_parquet(
            "wl2.parquet",
            {
                "sample_id": [0, 0, 0, 1, 1],
                "position": [0, 1, 2, 0, 1],
                "skill": [1, 2, 3, 4, 5],
                "question": [1, 2, 3, 4, 5],
                "response": [1, 1, 0, 0, 1],
                "mask": [0, 0, 1, 0, 1],
                "user_id": [7, 7, 7, 9, 9],
                "group_id": [1, 1, 1, 2, 2],
                "true_label": [1, 1, 0, 0, 1],
            },
        )
        dataset = WindowlateIterableDataset(path, max_seq_len=8)
        samples = list(iter(dataset))
        assert len(samples) == 2
        assert samples[0][-1].tolist() == [7, 7, 7, 0, 0, 0, 0, 0]
        assert samples[1][-1].tolist() == [9, 9, 0, 0, 0, 0, 0, 0]

    def test_len_counts_samples(
        self, write_windowlate_parquet: Callable[..., str]
    ) -> None:
        path = write_windowlate_parquet(
            "wl3.parquet",
            {
                "sample_id": [0, 0, 0, 1, 1],
                "position": [0, 1, 2, 0, 1],
                "skill": [1] * 5,
                "question": [1] * 5,
                "response": [1] * 5,
                "mask": [0, 0, 1, 0, 1],
                "user_id": [7, 7, 7, 9, 9],
                "group_id": [1] * 5,
                "true_label": [1] * 5,
            },
        )
        assert len(WindowlateIterableDataset(path, max_seq_len=8)) == 2


# --- SkillModelData.build_sequence_data ---------------------------------------


class TestBuildSequenceDataOriginalUser:
    """user_id_sequence fills from user, not the split row index."""

    @pytest.fixture
    def split_frame(self) -> pl.DataFrame:
        # Two original students (5 and 9), three splits between them:
        #   split row 0 -> original 5  (positions 0..2)
        #   split row 1 -> original 9  (positions 0..1)
        #   split row 2 -> original 5  (positions 0..1)
        return pl.DataFrame(
            {
                "sequence_id": [0, 0, 0, 1, 1, 2, 2],
                "seq_pos": [0, 1, 2, 0, 1, 0, 1],
                "skill": [10, 11, 12, 13, 14, 15, 16],
                "question": [1, 2, 3, 4, 5, 6, 7],
                "label": [1, 0, 1, 0, 1, 1, 0],
                "user": [5, 5, 5, 9, 9, 5, 5],
            }
        )

    def _make(self, frame: pl.DataFrame) -> SkillModelData:
        # Reuse the conftest factory pattern: concrete subclass + stub source.
        from tests.unit.model_data.conftest import StubDataSource

        class _Concrete(SkillModelData):
            def prepare_data(self, rc: RunConfig) -> None: ...

        return _Concrete(
            cast(DataSource, StubDataSource(frame, {"max_seq_len": 3, "num_users": 10}))
        )

    def test_user_id_sequence_carries_original_ids(
        self, split_frame: pl.DataFrame
    ) -> None:
        model_data = self._make(split_frame)
        _, _, _, user_id_sequence, _ = model_data.build_sequence_data()
        assert user_id_sequence.shape == (3, 3)
        # Split rows 0 and 2 belong to original student 5 -> same id shared.
        assert user_id_sequence[0].tolist() == [5, 5, 5]
        assert user_id_sequence[1].tolist() == [9, 9, 0]
        assert user_id_sequence[2].tolist() == [5, 5, 0]

    def test_missing_user_column_raises(self) -> None:
        # No backward compatibility: legacy split frames without the column
        # fail fast on data access.
        frame = pl.DataFrame(
            {
                "sequence_id": [0, 0],
                "seq_pos": [0, 1],
                "skill": [10, 11],
                "question": [1, 2],
                "label": [1, 0],
            }
        )
        model_data = self._make(frame)
        with pytest.raises(Exception):
            model_data.build_sequence_data()


# --- load_windowlate_data (numpy variant) -------------------------------------


class TestLoadWindowlateDataOrder:
    """user_id moved to the tuple tail, matching the iterable dataset order."""

    def _write_parquet(self, tmp_path: Path) -> str:
        path = tmp_path / "stub_windowlate.parquet"
        pq.write_table(
            pa.table(
                {
                    "sample_id": pa.array([0, 0, 0, 1, 1], pa.int32()),
                    "position": pa.array([0, 1, 2, 0, 1], pa.int32()),
                    "skill": pa.array([10, 11, 12, 13, 14], pa.int32()),
                    "question": pa.array([1, 2, 3, 4, 5], pa.int32()),
                    "response": pa.array([1, 0, 0, 0, 1], pa.int8()),
                    "mask": pa.array([0, 0, 1, 0, 1], pa.int8()),
                    "user_id": pa.array([7, 7, 7, 9, 9], pa.int32()),
                    "group_id": pa.array([1, 1, 1, 2, 2], pa.int64()),
                    "true_label": pa.array([1, 0, 1, 0, 1], pa.int8()),
                }
            ),
            path,
        )
        return str(path)

    def test_tuple_order(
        self, tmp_path: Path, make_skill_model_data: Callable[..., SkillModelData]
    ) -> None:
        import polars as pl

        path = self._write_parquet(tmp_path)
        lazy = pl.scan_parquet(path)
        model_data = make_skill_model_data(windowlate_data=lazy)
        out = model_data.load_windowlate_data(max_seq_len=4)
        assert len(out) == 7
        # user_id last, NOT at index 3 (its old numpy-variant position).
        user_id_sequence = out[-1]
        assert user_id_sequence.shape == (2, 4)
        assert user_id_sequence.dtype == np.int32
        assert user_id_sequence[0].tolist() == [7, 7, 7, 0]
        assert user_id_sequence[1].tolist() == [9, 9, 0, 0]
        # The other six keep the iterable dataset's order semantics.
        seq, _resp, mask, gid, _label, _q, _ = out
        assert seq[0].tolist() == [10, 11, 12, 0]
        assert mask[0].tolist() == [0, 0, 1, 0]
        assert gid[1].tolist() == [2, 2, -1, -1]

    def test_missing_windowlate_data_raises(
        self, make_skill_model_data: Callable[..., SkillModelData]
    ) -> None:
        model_data = make_skill_model_data(windowlate_data=None)
        with pytest.raises(ValueError):
            model_data.load_windowlate_data(max_seq_len=4)


# --- DataSource._build_split_sequences ----------------------------------------


class TestSplitPipelineOriginalUserColumn:
    """Split pipeline preserves the pre-split student id (user)."""

    @staticmethod
    def _build(frame: pl.DataFrame, max_seq_len: int, min_seq_len: int) -> pl.DataFrame:
        from utils.data_process.data_source import DataSource

        class _MinimalDS(DataSource):
            def load_src_data(self) -> None: ...
            def transform_data(self) -> None: ...
            def clean_raw_data(self) -> None: ...

        ds = _MinimalDS.__new__(_MinimalDS)
        ds.args = type(
            "Args", (), {"max_seq_len": max_seq_len, "min_seq_len": min_seq_len}
        )()
        ds.sequence_data = frame
        # expand_skills=False path never reads relation_data; None keeps
        # the fixture honest about the unprepared state.
        ds.relation_data = cast(Any, None)
        return ds._build_split_sequences(expand_skills=False)

    def test_user_column_present_and_matches_source(self) -> None:
        frame = pl.DataFrame(
            {
                "user": [0, 0, 0, 1, 1],
                "question": [1, 2, 3, 4, 5],
                "label": [1, 0, 1, 0, 1],
                "timestamp": [1, 2, 3, 1, 2],
            }
        )
        out = self._build(frame, max_seq_len=2, min_seq_len=1)
        assert "user" in out.columns
        assert "sequence_id" in out.columns
        # user 0 -> splits (0,0),(0,1) -> new ids 0,1; user 1 -> split id 2.
        # new ids are assigned per (user, split_idx) in global order.
        assert out["sequence_id"].to_list() == [0, 0, 1, 2, 2]
        assert out["user"].to_list() == [0, 0, 0, 1, 1]
        assert out["seq_pos"].to_list() == [0, 1, 0, 0, 1]

    def test_empty_input_keeps_schema(self) -> None:
        frame = pl.DataFrame(
            {
                "user": pl.Series([], dtype=pl.Int32),
                "question": pl.Series([], dtype=pl.Int32),
                "label": pl.Series([], dtype=pl.Int8),
                "timestamp": pl.Series([], dtype=pl.Int64),
            }
        )
        out = self._build(frame, max_seq_len=2, min_seq_len=1)
        assert "user" in out.columns
        assert "sequence_id" in out.columns
        assert "original_user" not in out.columns
        assert out.height == 0

    def test_min_seq_len_drop_keeps_global_id_order(self) -> None:
        # user 0: 4 interactions -> two full splits (ids 0,1); user 1: 1
        # interaction -> dropped by min_seq_len=2 -> user 2's single split
        # takes dense id 2 (ids stay dense, ordered by (user, split_idx)).
        frame = pl.DataFrame(
            {
                "user": [0, 0, 0, 0, 1, 2, 2],
                "question": [1, 2, 3, 4, 5, 6, 7],
                "label": [1, 0, 1, 0, 1, 1, 0],
                "timestamp": [1, 2, 3, 4, 1, 1, 2],
            }
        )
        out = self._build(frame, max_seq_len=2, min_seq_len=2)
        assert out["sequence_id"].to_list() == [0, 0, 1, 1, 2, 2]
        assert out["user"].to_list() == [0, 0, 0, 0, 2, 2]
        assert out["seq_pos"].to_list() == [0, 1, 0, 1, 0, 1]
