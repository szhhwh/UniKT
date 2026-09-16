"""Tests for DataSource._validate_data, _iter_user_aligned_slices, id remapping."""

from collections.abc import Callable, Sequence

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from utils.data_process.data_source import DataSource

# --- _validate_data (static) ---------------------------------------------------


class TestValidateData:
    """All guards raise AssertionError with the exact source messages."""

    @staticmethod
    def _relation_and_sequence(
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
        # User 0 sees questions 1/2, user 1 sees 2/3; relation covers 1/2/3.
        seq = make_sequence_frame(
            users=[0, 0, 1, 1], questions=[1, 2, 2, 3], labels=[1, 0, 1, 0]
        )
        rel = {"question_skill": make_question_skill_frame([(1, 10), (2, 11), (3, 12)])}
        return rel, seq

    def test_missing_question_skill_relation_raises(
        self, make_sequence_frame: Callable[..., pl.DataFrame]
    ) -> None:
        seq = make_sequence_frame(users=[0], questions=[1])
        with pytest.raises(AssertionError, match="question_skill relation is required"):
            DataSource._validate_data({}, seq)

    def test_question_skill_wrong_columns_raises(
        self,
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(users=[0], questions=[1])
        bad = make_question_skill_frame([(1, 10)]).with_columns(
            pl.lit(0, dtype=pl.Int32).alias("extra")
        )
        with pytest.raises(AssertionError, match="question_skill columns mismatch"):
            DataSource._validate_data({"question_skill": bad}, seq)

    def test_relation_not_exactly_two_columns_raises(
        self,
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(users=[0], questions=[1])
        rel = {
            "question_skill": make_question_skill_frame([(1, 10)]),
            "question_assignment": make_question_skill_frame([(1, 20)]).with_columns(
                pl.lit(0, dtype=pl.Int32).alias("slot")
            ),
        }
        with pytest.raises(AssertionError, match="should have exactly 2 columns, got"):
            DataSource._validate_data(rel, seq)

    def test_duplicate_relation_rows_raises(
        self,
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(users=[0], questions=[1])
        dup = pl.concat(
            [make_question_skill_frame([(1, 10)]), make_question_skill_frame([(1, 10)])]
        )
        with pytest.raises(AssertionError, match="has duplicate rows"):
            DataSource._validate_data({"question_skill": dup}, seq)

    def test_skill_column_in_sequence_raises(
        self,
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(users=[0], questions=[1]).with_columns(
            pl.lit(10, dtype=pl.Int32).alias("skill")
        )
        rel = {"question_skill": make_question_skill_frame([(1, 10)])}
        with pytest.raises(
            AssertionError, match="sequence_data should not contain 'skill' column"
        ):
            DataSource._validate_data(rel, seq)

    def test_sequence_question_missing_from_relation_raises(
        self,
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # question 3 answered in sequence but absent from the relation.
        seq = make_sequence_frame(users=[0, 0], questions=[1, 3])
        rel = {"question_skill": make_question_skill_frame([(1, 10)])}
        with pytest.raises(
            AssertionError, match="not found in question_skill relation"
        ):
            DataSource._validate_data(rel, seq)

    def test_relation_question_missing_from_sequence_raises(
        self,
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # question 3 in the relation but never answered in sequence.
        seq = make_sequence_frame(users=[0], questions=[1])
        rel = {"question_skill": make_question_skill_frame([(1, 10), (3, 12)])}
        with pytest.raises(
            AssertionError, match="question_skill has 2 unique, sequence_data has 1"
        ):
            DataSource._validate_data(rel, seq)

    def test_valid_data_passes(
        self,
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        rel, seq = self._relation_and_sequence(
            make_sequence_frame, make_question_skill_frame
        )
        assert DataSource._validate_data(rel, seq) is None


# --- _iter_user_aligned_slices ---------------------------------------------------


class TestIterUserAlignedSlices:
    @staticmethod
    def _frame(user_counts: Sequence[int]) -> pl.DataFrame:
        users = [u for u, n in enumerate(user_counts) for _ in range(n)]
        return pl.DataFrame({"user": pl.Series(users, dtype=pl.Int32)})

    def test_users_never_split_across_batches(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        # Row counts [2, 3, 4]: user0+user1 (5 rows) reach the target, user2
        # forms the trailing batch.
        df = self._frame([2, 3, 4])
        ds = make_data_source()
        slices = list(ds._iter_user_aligned_slices(df, 5))
        assert slices == [(0, 5), (5, 9)]
        # Each slice spans whole users and slices partition the frame.
        for start, end in slices:
            users_in_batch = df["user"][start:end].unique().to_list()
            assert len(users_in_batch) == df["user"][start:end].n_unique()
        assert slices[0][0] == 0 and slices[-1][1] == len(df)
        assert all(slices[i][1] == slices[i + 1][0] for i in range(len(slices) - 1))

    def test_many_small_users_pack_into_one_batch(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        # Five 1-row users accumulate to exactly the target -> single batch.
        df = self._frame([1, 1, 1, 1, 1])
        ds = make_data_source()
        assert list(ds._iter_user_aligned_slices(df, 5)) == [(0, 5)]

    def test_single_user_larger_than_limit_forms_one_oversized_batch(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        df = self._frame([7, 1])
        ds = make_data_source()
        assert list(ds._iter_user_aligned_slices(df, 3)) == [(0, 7), (7, 8)]

    def test_empty_input_yields_no_batches(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        df = pl.DataFrame({"user": pl.Series([], dtype=pl.Int32)})
        ds = make_data_source()
        assert list(ds._iter_user_aligned_slices(df, 5)) == []


# --- _remap_user_ids / _remap_question_ids ---------------------------------------


class TestRemapIds:
    def test_user_ids_remapped_dense_preserving_row_order(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        seq = pl.DataFrame(
            {
                "user": pl.Series([20, 20, 10, 10, 10], dtype=pl.Int32),
                "question": pl.Series([200, 100, 300, 100, 200], dtype=pl.Int32),
                "label": pl.Series([1, 0, 1, 1, 0], dtype=pl.Int8),
                "timestamp": pl.Series(range(5), dtype=pl.Int64),
            }
        )
        ds = make_data_source(sequence_data=seq)
        ds._remap_user_ids()
        out = ds.sequence_data
        # Sorted-unique users 10, 20 -> dense 0, 1; input row order kept.
        assert out["user"].to_list() == [1, 1, 0, 0, 0]
        assert out.schema["user"] == pl.Int32
        assert out["question"].to_list() == [200, 100, 300, 100, 200]

    def test_question_remap_three_step_consistency(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        seq = pl.DataFrame(
            {
                "user": pl.Series([20, 20, 10, 10, 10], dtype=pl.Int32),
                "question": pl.Series([200, 100, 300, 100, 200], dtype=pl.Int32),
                "label": pl.Series([1, 0, 1, 1, 0], dtype=pl.Int8),
                "timestamp": pl.Series(range(5), dtype=pl.Int64),
            }
        )
        rel = {
            "question_skill": pl.DataFrame(
                {
                    "question": pl.Series([100, 100, 200, 999], dtype=pl.Int32),
                    "skill": pl.Series([50, 51, 52, 53], dtype=pl.Int32),
                }
            )
        }
        ds = make_data_source(sequence_data=seq, relation_data=rel)
        ds._remap_user_ids()
        ds._remap_question_ids()

        # Step 1+2: active questions {100, 200, 300} -> dense {0, 1, 2};
        # question 999 (relation-only) is filtered out.
        assert sorted(ds.sequence_data["question"].unique().to_list()) == [0, 1, 2]
        qs = ds.relation_data["question_skill"]
        assert sorted(qs["question"].unique().to_list()) == [0, 1]
        assert qs.height == 3  # the (999, 53) row is gone

        # Step 3: skill ids remapped dense independently per relation.
        assert sorted(qs["skill"].unique().to_list()) == [0, 1, 2]

        # Sequence and relation agree on the shared question mapping.
        seq_pairs = set(zip(seq["question"], ds.sequence_data["question"]))
        expected = {(100, 0), (200, 1), (300, 2)}
        assert expected <= seq_pairs
        rel_pairs = set(
            zip(
                [100, 100, 200],
                qs.sort("skill")["question"].to_list(),
            )
        )
        assert rel_pairs == {(100, 0), (200, 1)}

    def test_remap_deterministic_across_calls(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        seq = pl.DataFrame(
            {
                "user": pl.Series([7, 7, 3], dtype=pl.Int32),
                "question": pl.Series([50, 60, 50], dtype=pl.Int32),
                "label": pl.Series([1, 0, 1], dtype=pl.Int8),
                "timestamp": pl.Series(range(3), dtype=pl.Int64),
            }
        )
        rel = {
            "question_skill": pl.DataFrame(
                {
                    "question": pl.Series([50, 60], dtype=pl.Int32),
                    "skill": pl.Series([9, 8], dtype=pl.Int32),
                }
            )
        }
        ds1 = make_data_source(sequence_data=seq, relation_data=rel)
        ds1._remap_user_ids()
        ds1._remap_question_ids()

        ds2 = make_data_source(sequence_data=seq, relation_data=rel)
        ds2._remap_user_ids()
        ds2._remap_question_ids()
        assert_frame_equal(ds1.sequence_data, ds2.sequence_data)
        assert_frame_equal(
            ds1.relation_data["question_skill"], ds2.relation_data["question_skill"]
        )

        # Idempotence: ids are already dense, a second pass is a no-op.
        before = ds1.sequence_data.clone()
        ds1._remap_user_ids()
        ds1._remap_question_ids()
        assert_frame_equal(before, ds1.sequence_data)
