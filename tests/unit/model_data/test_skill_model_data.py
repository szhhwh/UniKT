"""Tests for ``utils.model_data.skill_model_data``.

Two concerns:

- ``WindowlateIterableDataset`` / ``SkillModelData`` preserve the original
  student id across subsequences and fold partitions.
- ``create_windowlate_dataloader`` is batch-equivalent to a plain
  ``DataLoader`` over the same dataset, across worker counts, drop_last,
  spawn contexts and persistent workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from utils.config.data_config import DataLoaderConfig, create_optimized_dataloader
from utils.data_process import DataSource
from utils.model_data.skill_model_data import (
    SkillModelData,
    WindowlateIterableDataset,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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


# --- Packed sequence identity ------------------------------------------------


class TestPackedSequenceOriginalUser:
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
                "fold": [0, 0, 0, -1, -1, 0, 0],
            }
        )

    def _make(self, frame: pl.DataFrame) -> SkillModelData:
        # Reuse the conftest factory pattern: concrete subclass + stub source.
        from tests.unit.model_data.conftest import StubDataSource

        class _Concrete(SkillModelData):
            def prepare_data(self, rc: RunConfig) -> None: ...

        return _Concrete(
            cast(
                DataSource,
                StubDataSource(
                    frame,
                    {"max_skill_seq_len": 3, "num_users": 10, "kfold_n_splits": 2},
                ),
            )
        )

    def test_user_id_sequence_carries_original_ids(
        self, split_frame: pl.DataFrame
    ) -> None:
        model_data = self._make(split_frame)
        sequences = model_data.load_sequences()
        assert sequences.user.tolist() == [5, 9, 5]
        assert sequences.sequence_id.tolist() == [0, 1, 2]
        assert sequences.fold.tolist() == [0, -1, 0]
        assert sequences.mask.tolist() == [
            [True, True, True],
            [True, True, False],
            [True, True, False],
        ]

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
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            model_data.load_sequences()


# --- create_windowlate_dataloader: batch-equivalence with a plain DataLoader ---


@pytest.fixture
def windowlate_path(tmp_path: Path) -> str:
    # Multi-skill targets share histories/group ids; lengths include singleton,
    # short, full and expanded windows. Row groups contain 2/1/3/3 samples.
    lengths = [1, 2, 2, 4, 4, 6, 6, 8, 8]
    groups = [101, 205, 205, 309, 309, 411, 411, 512, 610]
    labels = [1, 0, 0, 1, 1, 0, 0, 1, 1]
    samples = []
    for index, (length, group, label) in enumerate(zip(lengths, groups, labels)):
        positions = np.arange(length)
        skill = (positions % 3 + 1).astype(np.int32)
        skill[-1] = index % 7 + 1
        response = (positions % 2).astype(np.int8)
        response[-1] = 0
        true_label = response.copy()
        true_label[-1] = label
        samples.append(
            pa.table(
                {
                    "sample_id": np.full(length, index * 3 + 11, dtype=np.int64),
                    "position": positions.astype(np.int32),
                    "skill": skill,
                    "question": (positions + 1).astype(np.int32),
                    "response": response,
                    "mask": (positions == length - 1).astype(np.int8),
                    "user_id": np.full(length, 7 if index < 5 else 42, dtype=np.int32),
                    "group_id": np.full(length, group, dtype=np.int64),
                    "true_label": true_label,
                    "timestamp": (positions * 60_000 + 1_000).astype(np.int64),
                }
            )
        )
    path = str(tmp_path / "stub_windowlate.parquet")
    with pq.ParquetWriter(path, samples[0].schema) as writer:
        for start, end in ((0, 2), (2, 3), (3, 6), (6, 9)):
            writer.write_table(pa.concat_tables(samples[start:end]))
    return path


@pytest.fixture
def windowlate_model_data(
    windowlate_path: str, make_skill_model_data: Callable[..., SkillModelData]
) -> SkillModelData:
    model_data = make_skill_model_data(
        metadata={
            "max_windowlate_seq_len": 8,
            "max_skill_seq_len": 8,
            "kfold_n_splits": 2,
        }
    )
    model_data.data_src.data_folder = str(Path(windowlate_path).parent)
    return model_data


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("drop_last", [False, True])
def test_batches_match_legacy_loader(
    windowlate_model_data: SkillModelData,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
) -> None:
    dataset = windowlate_model_data._create_windowlate_dataset()
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": drop_last,
    }
    legacy = list(DataLoader(dataset, **kwargs))
    optimized_loader = windowlate_model_data.create_windowlate_dataloader(**kwargs)
    optimized = list(optimized_loader)
    assert len(optimized_loader) == len(DataLoader(dataset, **kwargs))
    assert len(optimized) == len(legacy)
    for actual, expected in zip(optimized, legacy, strict=True):
        assert type(actual) is type(expected)
        for actual_field, expected_field in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)
    # Constructing a loader must not switch the source's public iterator to raw views.
    assert isinstance(next(iter(dataset)), tuple)


def test_batch_path_skips_single_sample_tensor_builder(
    windowlate_model_data: SkillModelData, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise AssertionError("The batched path must not allocate per-sample tensors")

    monkeypatch.setattr(WindowlateIterableDataset, "_build_single_tensor", fail)
    batch = next(iter(windowlate_model_data.create_windowlate_dataloader(batch_size=4)))
    assert len(batch) == 7
    assert batch[0].shape == (4, 8)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_public_loader_options(
    windowlate_model_data: SkillModelData, num_workers: int
) -> None:
    loader = windowlate_model_data.create_windowlate_dataloader(
        batch_size=4,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        drop_last=True,
        timeout=0,
    )
    assert loader.batch_size == 4
    assert loader.num_workers == num_workers
    assert loader.pin_memory is True
    assert loader.drop_last is True
    assert loader.prefetch_factor == (3 if num_workers else None)
    assert loader.persistent_workers is bool(num_workers)


def test_shared_factory_uses_batch_construction(windowlate_path: str) -> None:
    dataset = WindowlateIterableDataset(windowlate_path, max_seq_len=8)
    loader = create_optimized_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        config=DataLoaderConfig(num_workers=0),
        device=torch.device("cpu"),
    )
    assert loader.collate_fn == dataset._collate_batch
    assert next(iter(loader))[0].shape == (4, 8)


def test_spawn_workers_match_legacy_loader(
    windowlate_model_data: SkillModelData,
) -> None:
    dataset = windowlate_model_data._create_windowlate_dataset()
    kwargs: dict[str, Any] = {
        "batch_size": 4,
        "num_workers": 2,
        "multiprocessing_context": "spawn",
    }
    for actual, expected in zip(
        windowlate_model_data.create_windowlate_dataloader(**kwargs),
        DataLoader(dataset, **kwargs),
        strict=True,
    ):
        for actual_field, expected_field in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)


def test_persistent_workers_preserve_repeated_iteration(
    windowlate_model_data: SkillModelData,
) -> None:
    dataset = windowlate_model_data._create_windowlate_dataset()
    kwargs: dict[str, Any] = {
        "batch_size": 4,
        "num_workers": 2,
        "persistent_workers": True,
    }
    legacy = DataLoader(dataset, **kwargs)
    optimized = windowlate_model_data.create_windowlate_dataloader(**kwargs)
    for _ in range(2):
        for actual, expected in zip(optimized, legacy, strict=True):
            for actual_field, expected_field in zip(actual, expected, strict=True):
                torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)
