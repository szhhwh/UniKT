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
        # The first interaction provides history; the other skills are targets.
        assert len(samples) == 3
        assert samples == [
            # Second interaction, target skill 101: 1 history row + target.
            [(0, 0, 100, 10, 1, 0, 7, 0, 1), (0, 1, 101, 11, 0, 1, 7, 1, 0)],
            # Target skill 102 of the same interaction reuses the same history.
            [(1, 0, 100, 10, 1, 0, 7, 0, 1), (1, 1, 102, 11, 0, 1, 7, 1, 0)],
            [
                (2, 0, 100, 10, 1, 0, 7, 0, 1),
                (2, 1, 101, 11, 0, 0, 7, 1, 0),
                (2, 2, 102, 11, 0, 0, 7, 1, 0),
                (2, 3, 103, 12, 0, 1, 7, 2, 1),
            ],
        ]

    def test_second_user_id_offsets_increment(self, processor_defaults: None) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                user_id=8,
                labels=[0, 1],
                skills_list=[[100], [104, 105]],
                questions=[10, 13],
                sample_id_start=3,
                group_id_start=3,
                max_seq_len=100,
                extras={},
            )
        )
        assert len(samples) == 2
        assert samples[0] == [
            (3, 0, 100, 10, 0, 0, 8, 3, 0),
            (3, 1, 104, 13, 0, 1, 8, 4, 1),
        ]
        assert samples[1] == [
            (4, 0, 100, 10, 0, 0, 8, 3, 0),
            (4, 1, 105, 13, 0, 1, 8, 4, 1),
        ]

    def test_history_keeps_true_response_target_zeroed(
        self, processor_defaults: None
    ) -> None:
        # History responses retain their labels; the target response is zero.
        samples = list(
            WindowlateProcessor.generate_user_samples(
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=100,
                **_U7,
            )
        )
        target_sample = samples[2]  # history [1, 0, 0], target label 1
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
        # Samples 0..1 fit within 2 rows; sample 2's 4-row
        # history+target keeps only the last 2 rows.
        assert samples[0] == [
            (0, 0, 100, 10, 1, 0, 7, 0, 1),
            (0, 1, 101, 11, 0, 1, 7, 1, 0),
        ]
        assert samples[2] == [
            (2, 0, 102, 11, 0, 0, 7, 1, 0),
            (2, 1, 103, 12, 0, 1, 7, 2, 1),
        ]

    def test_interaction_window_keeps_whole_interactions(
        self, processor_defaults: None
    ) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=2,
                truncation_stage="pre_expansion",
                **_U7,
            )
        )
        # Sample 1 (target 102; the history never carries its sibling skill
        # 101) stays within the budget either way.
        assert samples[1] == [
            (1, 0, 100, 10, 1, 0, 7, 0, 1),
            (1, 1, 102, 11, 0, 1, 7, 1, 0),
        ]
        # Sample 2 (target 103) holds the last 2 whole interactions: the
        # two-skill interaction 1 stays complete (3 rows); counting expanded
        # skills would keep only [102, 103].
        assert samples[2] == [
            (2, 0, 101, 11, 0, 0, 7, 1, 0),
            (2, 1, 102, 11, 0, 0, 7, 1, 0),
            (2, 2, 103, 12, 0, 1, 7, 2, 1),
        ]
        assert max(len(s) for s in samples) == 3

    @pytest.mark.parametrize("stage", ["post_expansion", "pre_expansion"])
    def test_multiskill_first_question_is_history_only(
        self, processor_defaults: None, stage: str
    ) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                user_id=7,
                labels=[1, 0],
                skills_list=[[100, 101], [102, 103]],
                questions=[10, 11],
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=3,
                extras={},
                truncation_stage=stage,
            )
        )
        assert len(samples) == 2
        assert [rows[-1][2] for rows in samples] == [102, 103]
        assert [rows[-1][7] for rows in samples] == [1, 1]
        for rows in samples:
            assert [row[2] for row in rows[:-1]] == [100, 101]
            assert [row[4] for row in rows[:-1]] == [1, 1]
            assert [row[5] for row in rows] == [0, 0, 1]
            assert [row[7] for row in rows[:-1]] == [0, 0]
            assert rows[-1][4] == rows[-1][8] == 0

    @pytest.mark.parametrize("stage", ["post_expansion", "pre_expansion"])
    @pytest.mark.parametrize("skills", [[100], [100, 101]])
    def test_single_interaction_yields_no_samples(
        self, processor_defaults: None, stage: str, skills: list[int]
    ) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                user_id=7,
                labels=[1],
                skills_list=[skills],
                questions=[10],
                sample_id_start=0,
                group_id_start=0,
                max_seq_len=100,
                extras={},
                truncation_stage=stage,
            )
        )
        assert samples == []

    def test_leading_empty_interaction_has_no_skill_history(
        self, processor_defaults: None
    ) -> None:
        samples = list(
            WindowlateProcessor.generate_user_samples(
                user_id=7,
                labels=[0, 1, 0],
                skills_list=[[], [100, 101], [102]],
                questions=[9, 10, 11],
                sample_id_start=9,
                group_id_start=5,
                max_seq_len=100,
                extras={},
            )
        )
        assert samples == [
            [
                (9, 0, 100, 10, 1, 0, 7, 6, 1),
                (9, 1, 101, 10, 1, 0, 7, 6, 1),
                (9, 2, 102, 11, 0, 1, 7, 7, 0),
            ]
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
        assert WindowlateProcessor.count_user_samples(skills_list, 1) == 3
        assert WindowlateProcessor.count_user_samples(skills_list, 100) == 3

    def test_long_user_counts_fully(self) -> None:
        # Nine targets follow the first interaction.
        skills_list = [[5]] * 10
        assert WindowlateProcessor.count_user_samples(skills_list, max_seq_len=2) == 9
        assert WindowlateProcessor.count_user_samples([], max_seq_len=2) == 0

    @pytest.mark.parametrize(
        ("skills_list", "expected"),
        [
            ([[]], 0),
            ([[1, 2]], 0),
            ([[], [1, 2], [3, 4]], 2),
            ([[1], [], [2]], 1),
        ],
    )
    def test_count_excludes_targets_without_history(
        self, skills_list: list[list[int]], expected: int
    ) -> None:
        assert WindowlateProcessor.count_user_samples(skills_list, 100) == expected


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
        # The 2-row and 3-row samples share a flush; each larger sample flushes.
        batch_idx, path, rows, max_window = WindowlateProcessor.process_user_batch(
            (0, [self._user(1, 10)], 100, 3, str(tmp_path), "post_expansion")
        )
        assert batch_idx == 0
        assert rows == 54  # 2+3+...+10
        assert max_window == 10
        assert pq.ParquetFile(path).num_row_groups == 8

    def test_below_limit_single_row_group(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        _, path, rows, _ = WindowlateProcessor.process_user_batch(
            (0, [self._user(1, 3)], 100, 500_000, str(tmp_path), "post_expansion")
        )
        assert rows == 5  # 2+3
        assert pq.ParquetFile(path).num_row_groups == 1

    def test_empty_batch_returns_none_and_writes_nothing(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        result = WindowlateProcessor.process_user_batch(
            (1, [], 100, 3, str(tmp_path), "post_expansion")
        )
        assert result == (1, None, 0, 0)
        assert list(tmp_path.iterdir()) == []

    def test_single_interaction_user_writes_nothing(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        result = WindowlateProcessor.process_user_batch(
            (0, [self._user(1, 1)], 100, 3, str(tmp_path), "post_expansion")
        )
        assert result == (0, None, 0, 0)
        assert list(tmp_path.iterdir()) == []

    def test_flushed_columns_match_core_schema(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        _, path, _, _ = WindowlateProcessor.process_user_batch(
            (0, [self._user(1, 2)], 100, 500_000, str(tmp_path), "post_expansion")
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
        assert df["fold"].to_list() == [-1, -1]


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
        # User 7's second interaction supplies two targets; user 8 has no history.
        assert df.rows() == [
            (0, 0, 100, 10, 1, 0, 7, 0, 1, -1, 1),
            (0, 1, 101, 11, 0, 1, 7, 1, 0, -1, 2),
            (1, 0, 100, 10, 1, 0, 7, 0, 1, -1, 1),
            (1, 1, 102, 11, 0, 1, 7, 1, 0, -1, 2),
        ]

    @pytest.mark.parametrize("num_workers", [1, 2])
    def test_sample_ids_are_contiguous_across_users(
        self, processor_defaults: None, tmp_path: Path, num_workers: int
    ) -> None:
        test_data = pl.DataFrame(
            {
                "user": pl.Series([6, 7, 7, 8, 9, 9], dtype=pl.Int32),
                "question": pl.Series([10, 10, 11, 10, 11, 10], dtype=pl.Int32),
                "label": pl.Series([0, 1, 0, 1, 0, 1], dtype=pl.Int8),
                "timestamp": pl.Series([1, 1, 2, 1, 1, 2], dtype=pl.Int64),
            }
        )
        output_path = str(tmp_path / "windowlate.parquet")
        max_window = WindowlateProcessor.build(
            test_data=test_data,
            question_data=self._question_data(),
            max_seq_len=100,
            output_path=output_path,
            num_workers=num_workers,
            users_per_batch=1,
        )
        df = pl.read_parquet(output_path)
        targets = df.filter(pl.col("mask") == 1)
        assert df["sample_id"].to_list() == [0, 0, 1, 1, 2, 2, 2]
        assert targets["user_id"].to_list() == [7, 7, 9]
        assert targets["group_id"].to_list() == [2, 2, 5]
        assert targets["skill"].to_list() == [101, 102, 100]
        assert targets["true_label"].to_list() == [0, 0, 1]
        assert targets["position"].to_list() == [1, 1, 2]
        assert max_window == 3
        last_history = df.filter((pl.col("sample_id") == 2) & (pl.col("mask") == 0))
        assert last_history["skill"].to_list() == [101, 102]
        assert last_history["group_id"].to_list() == [4, 4]

    def test_build_returns_max_window_len(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        # Three interactions, the middle one carrying two skills: with the
        # target at the last interaction, post_expansion keeps its last 2 rows
        # while pre_expansion keeps the middle interaction whole (3 rows).
        test_data = pl.DataFrame(
            {
                "user": pl.Series([7, 7, 7], dtype=pl.Int32),
                "question": pl.Series([10, 11, 12], dtype=pl.Int32),
                "label": pl.Series([1, 0, 1], dtype=pl.Int8),
                "timestamp": pl.Series([1, 2, 3], dtype=pl.Int64),
            }
        )
        question_data = self._question_data().vstack(
            pl.DataFrame(
                {
                    "question": pl.Series([12], dtype=pl.Int32),
                    "skill": pl.Series([103], dtype=pl.Int32),
                }
            )
        )
        for stage, expected in (("post_expansion", 2), ("pre_expansion", 3)):
            output_path = str(tmp_path / f"windowlate_{stage}.parquet")
            max_window = WindowlateProcessor.build(
                test_data=test_data,
                question_data=question_data,
                max_seq_len=2,
                output_path=output_path,
                num_workers=1,
                users_per_batch=1,
                truncation_stage=stage,
            )
            assert max_window == expected, stage

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

    def test_only_first_interactions_raise_without_output(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        test_data = self._test_data().filter(pl.col("timestamp") == 1)
        output_path = str(tmp_path / "windowlate.parquet")
        with pytest.raises(
            ValueError, match="No valid windowlate evaluation samples for test set"
        ):
            WindowlateProcessor.build(
                test_data=test_data,
                question_data=self._question_data(),
                max_seq_len=100,
                output_path=output_path,
                num_workers=1,
            )
        assert not os.path.exists(output_path)


# --- _merge_results ---------------------------------------------------------------------


class TestMergeResults:
    def test_merge_is_atomic_replace(
        self, processor_defaults: None, tmp_path: Path
    ) -> None:
        work = tmp_path / "work"
        work.mkdir()
        # user_a has two targets, each with one history row.
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
            [0, 1],
            [[12], [13]],
            [7, 8],
            2,
            2,
            {},
        )
        res_a = WindowlateProcessor.process_user_batch(
            (0, [user_a], 10, 500_000, str(work), "post_expansion")
        )
        res_b = WindowlateProcessor.process_user_batch(
            (1, [user_b], 10, 500_000, str(work), "post_expansion")
        )

        output_path = str(tmp_path / "merged.parquet")
        total, max_window = WindowlateProcessor._merge_results(
            [res_a, res_b], output_path
        )

        assert total == res_a[2] + res_b[2]
        assert max_window == max(res_a[3], res_b[3])
        assert os.path.exists(output_path)
        # os.replace consumed the staging file; nothing scratch is left behind.
        assert not os.path.exists(output_path + ".tmp")
        df = pl.read_parquet(output_path)
        # Worker order preserved: all of user_a's samples precede user_b's.
        assert df["user_id"].to_list() == [1, 1, 1, 1, 2, 2]
        assert df["sample_id"].to_list() == [0, 0, 1, 1, 2, 2]

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
