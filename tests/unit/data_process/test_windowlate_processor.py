"""Tests for WindowlateProcessor: sample generation, batching, flush, merge.

Hand-computed golden tuples for generate_user_samples: two users, five
interactions total. Tuple order is
``(sample_id, position, skill, question, response, mask, user_id, group_id,
true_label)``.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow.parquet as pq
import pytest

from utils.data_process.windowlate_processor import WindowlateProcessor

_U7: dict[str, Any] = {
    "user_id": 7,
    "labels": [1, 0, 1],
    "skills_list": [[100], [101, 102], [103]],
    "questions": [10, 11, 12],
    "extras": {},
}


@pytest.fixture
def processor_defaults() -> Iterator[None]:
    """Restore the class-level column config: build()/_init_worker mutate it."""
    saved_cols = WindowlateProcessor.EXTRA_COLUMNS
    saved_dtypes = WindowlateProcessor.EXTRA_DTYPES
    yield
    WindowlateProcessor.EXTRA_COLUMNS = saved_cols
    WindowlateProcessor.EXTRA_DTYPES = saved_dtypes


# --- generate_user_samples ---------------------------------------------------------


class TestGenerateUserSamples:
    def test_first_user_exact_rows(self, processor_defaults: None) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=100,
                **_U7,
            )
        )
        # 3 interactions -> 1 + 2 + 1 samples (one per target skill).
        assert len(samples) == 4
        assert samples == [
            # First interaction has no history: a single masked target row.
            [(0, 0, 100, 10, 0, 1, 7, 0, 1)],
            # Second interaction, target skill 101: 1 history row + target.
            [(1, 0, 100, 10, 1, 0, 7, 0, 1), (1, 1, 101, 11, 0, 1, 7, 1, 0)],
            # Target skill 102 of the same interaction reuses the same history.
            [(2, 0, 100, 10, 1, 0, 7, 0, 1), (2, 1, 102, 11, 0, 1, 7, 1, 0)],
            [
                (3, 0, 100, 10, 1, 0, 7, 0, 1),
                (3, 1, 101, 11, 0, 0, 7, 1, 0),
                (3, 2, 102, 11, 0, 0, 7, 1, 0),
                (3, 3, 103, 12, 0, 1, 7, 2, 1),
            ],
        ]

    def test_second_user_id_offsets_increment(self, processor_defaults: None) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                user_id=8,
                labels=[0, 1],
                skills_list=[[100], [104, 105]],
                questions=[10, 13],
                sample_id_start=4,
                group_id_start=3,
                max_seq_len=100,
                extras={},
            )
        )
        assert len(samples) == 3
        assert samples[0] == [(4, 0, 100, 10, 0, 1, 8, 3, 0)]
        assert samples[1] == [
            (5, 0, 100, 10, 0, 0, 8, 3, 0),
            (5, 1, 104, 13, 0, 1, 8, 4, 1),
        ]
        assert samples[2] == [
            (6, 0, 100, 10, 0, 0, 8, 3, 0),
            (6, 1, 105, 13, 0, 1, 8, 4, 1),
        ]

    def test_history_keeps_true_response_target_zeroed(
        self, processor_defaults: None
    ) -> None:
        # NOTE: pinned current behavior -- leakage prevention zeroes the
        # response of the TARGET row (the answer being predicted), while
        # history rows keep their true labels in `response`; the target's real
        # answer lives only in `true_label`.
        samples = list(
            WindowlateProcessor.generate_user_samples(
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=100,
                **_U7,
            )
        )
        target_sample = samples[3]  # history [1, 0, 0], target label 1
        responses = [row[4] for row in target_sample]
        true_labels = [row[8] for row in target_sample]
        assert responses == [1, 0, 0, 0]
        assert true_labels == [1, 0, 0, 1]

    def test_window_truncated_to_last_max_seq_len(
        self, processor_defaults: None
    ) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=2,
                **_U7,
            )
        )
        # Samples 0..2 fit within 2 rows and stay intact; sample 3's 4-row
        # history+target keeps only the last 2 rows.
        assert samples[0] == [(0, 0, 100, 10, 0, 1, 7, 0, 1)]
        assert samples[1] == [
            (1, 0, 100, 10, 1, 0, 7, 0, 1),
            (1, 1, 101, 11, 0, 1, 7, 1, 0),
        ]
        assert samples[3] == [
            (3, 0, 102, 11, 0, 0, 7, 1, 0),
            (3, 1, 103, 12, 0, 1, 7, 2, 1),
        ]

    def test_mask_one_only_at_target_position(self, processor_defaults: None) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=100,
                **_U7,
            )
        )
        for rows in samples:
            masks = [row[5] for row in rows]
            positions = [row[1] for row in rows]
            assert masks == [0] * (len(rows) - 1) + [1]
            assert positions == list(range(len(rows)))

    def test_empty_skills_yield_nothing(self, processor_defaults: None) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                user_id=1,
                labels=[1, 0],
                skills_list=[[], []],
                questions=[10, 11],
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=10,
                extras={},
            )
        )
        assert samples == []


# --- count_user_samples ------------------------------------------------------------


class TestCountUserSamples:
    def test_count_ignores_max_seq_len(self) -> None:
        skills_list = [[1], [2, 3], [4]]
        assert WindowlateProcessor.count_user_samples(skills_list, 1) == 4
        assert WindowlateProcessor.count_user_samples(skills_list, 100) == 4

    def test_long_user_counts_fully(self) -> None:
        # 10 single-skill interactions -> 10 samples whatever the window size.
        skills_list = [[5]] * 10
        assert WindowlateProcessor.count_user_samples(skills_list, max_seq_len=2) == 10
        assert WindowlateProcessor.count_user_samples([], max_seq_len=2) == 0


# --- process_user_batch / _flush_buffers --------------------------------------------


class TestProcessUserBatch:
    @staticmethod
    def _user(
        user_id: int, n_interactions: int
    ) -> tuple[int, list[int], list[list[int]], list[int], int, int, dict[str, Any]]:
        return (
            user_id,
            [1] * n_interactions,
            [[5]] * n_interactions,
            [3] * n_interactions,
            0,  # sample_id_start
            0,  # group_id_start
            {},
        )

    def test_flush_triggered_by_chunk_row_limit(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        # Sample i has i+1 rows (history grows), so with limit 3 the buffer
        # flushes after samples 2..10 -> 1 leftover group + 8 mid flushes.
        batch_idx, path, rows = WindowlateProcessor.process_user_batch(
            (0, [self._user(1, 10)], 100, 3, str(tmp_path))
        )
        assert batch_idx == 0
        assert rows == 55  # 1+2+...+10
        assert pq.ParquetFile(path).num_row_groups == 9

    def test_below_limit_single_row_group(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        _, path, rows = WindowlateProcessor.process_user_batch(
            (0, [self._user(1, 3)], 100, 500_000, str(tmp_path))
        )
        assert rows == 6  # 1+2+3
        assert pq.ParquetFile(path).num_row_groups == 1

    def test_empty_batch_returns_none_and_writes_nothing(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        result = WindowlateProcessor.process_user_batch((1, [], 100, 3, str(tmp_path)))
        assert result == (1, None, 0)
        assert list(tmp_path.iterdir()) == []

    def test_flushed_columns_match_core_schema(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        _, path, _ = WindowlateProcessor.process_user_batch(
            (0, [self._user(1, 2)], 100, 500_000, str(tmp_path))
        )
        # A flush is asserted above via num_row_groups, so path is not None.
        df = pl.read_parquet(cast("str", path))
        assert df.columns == [
            "sample_id",
            "position",
            "skill",
            "question",
            "response",
            "mask",
            "user_id",
            "group_id",
            "true_label",
            "fold",
        ]
        # fold is a filled constant, not buffered per row.
        assert df["fold"].to_list() == [-1, -1, -1]


# --- build ----------------------------------------------------------------------------


class TestBuild:
    @staticmethod
    def _test_data() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "user": pl.Series([7, 7, 8], dtype=pl.Int32),
                "question": pl.Series([10, 11, 10], dtype=pl.Int32),
                "label": pl.Series([1, 0, 0], dtype=pl.Int8),
                "timestamp": pl.Series([1, 2, 1], dtype=pl.Int64),
            }
        )

    @staticmethod
    def _question_data() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "question": pl.Series([10, 11, 11], dtype=pl.Int32),
                "skill": pl.Series([100, 101, 102], dtype=pl.Int32),
            }
        )

    def test_end_to_end_rows_and_extra_columns(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        output_path = str(tmp_path / "windowlate.parquet")
        WindowlateProcessor.build(
            test_data=self._test_data(),
            question_data=self._question_data(),
            max_seq_len=100,
            output_path=output_path,
            num_workers=1,
            users_per_batch=1,
        )
        df = pl.read_parquet(output_path)
        # timestamp is not a core column -> preserved as an extra, after fold.
        assert df.columns == [
            "sample_id",
            "position",
            "skill",
            "question",
            "response",
            "mask",
            "user_id",
            "group_id",
            "true_label",
            "fold",
            "timestamp",
        ]
        # User 7 yields 3 samples (1 + 2 skills), user 8 one -> 6 rows total.
        # group ids advance by interaction count: user 8's rows carry group 2.
        assert df.rows() == [
            (0, 0, 100, 10, 0, 1, 7, 0, 1, -1, 1),
            (1, 0, 100, 10, 1, 0, 7, 0, 1, -1, 1),
            (1, 1, 101, 11, 0, 1, 7, 1, 0, -1, 2),
            (2, 0, 100, 10, 1, 0, 7, 0, 1, -1, 1),
            (2, 1, 102, 11, 0, 1, 7, 1, 0, -1, 2),
            (3, 0, 100, 10, 0, 1, 8, 2, 0, -1, 1),
        ]

    def test_all_questions_missing_raises(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        test_data = self._test_data().filter(pl.col("user") == 7)
        # Inner join on question drops everything -> no evaluable samples.
        question_data = self._question_data().filter(pl.col("question") == 999)
        with pytest.raises(
            ValueError, match="No valid windowlate evaluation samples for test set"
        ):
            WindowlateProcessor.build(
                test_data=test_data,
                question_data=question_data,
                max_seq_len=10,
                output_path=str(tmp_path / "out.parquet"),
                num_workers=1,
            )


# --- _merge_results ---------------------------------------------------------------------


class TestMergeResults:
    def test_merge_is_atomic_replace(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        # user_a: 1+2 target skills -> samples 0 (1 row), 1 and 2 (2 rows each).
        user_a: tuple[
            int, list[int], list[list[int]], list[int], int, int, dict[str, Any]
        ] = (
            1,
            [1, 0],
            [[10], [11, 12]],
            [5, 6],
            0,
            0,
            {},
        )
        user_b: tuple[
            int, list[int], list[list[int]], list[int], int, int, dict[str, Any]
        ] = (
            2,
            [1],
            [[12]],
            [7],
            3,
            2,
            {},
        )
        res_a = WindowlateProcessor.process_user_batch(
            (0, [user_a], 10, 500_000, str(work))
        )
        res_b = WindowlateProcessor.process_user_batch(
            (1, [user_b], 10, 500_000, str(work))
        )

        output_path = str(tmp_path / "merged.parquet")
        total = WindowlateProcessor._merge_results([res_a, res_b], output_path)

        assert total == res_a[2] + res_b[2]
        assert os.path.exists(output_path)
        # os.replace consumed the staging file; nothing scratch is left behind.
        assert not os.path.exists(output_path + ".tmp")
        df = pl.read_parquet(output_path)
        # Worker order preserved: all of user_a's samples precede user_b's.
        assert df["user_id"].to_list() == [1, 1, 1, 1, 1, 2]
        assert df["sample_id"].to_list() == [0, 1, 1, 2, 2, 3]

    def test_no_written_rows_raises_without_output(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        output_path = str(tmp_path / "merged.parquet")
        with pytest.raises(
            ValueError, match="No valid windowlate evaluation samples generated"
        ):
            WindowlateProcessor._merge_results([], output_path)
        assert not os.path.exists(output_path)
        assert not os.path.exists(output_path + ".tmp")
