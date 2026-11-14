"""Tests for windowlate sample identity, shapes, padding, and dtypes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from utils.model_data.windowlate import WindowlateIterableDataset

if TYPE_CHECKING:
    from collections.abc import Callable


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
