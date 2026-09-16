"""Tests for DataSource split-sequence building, build_* guards, add_kfold_labels."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from utils.data_process.data_source import DataSource

# Three users, each answering questions [1, 2, 3] whose skills expand to
# [10], [11, 12], [13] -> every user has 4 expanded skill rows.
_USERS = [0] * 3 + [1] * 3 + [2] * 3
_QUESTIONS = [1, 2, 3] * 3
_LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1]
_TIMESTAMPS = [1, 2, 3, 1, 2, 3, 1, 2, 3]
_QS_PAIRS = [(1, 10), (2, 11), (2, 12), (3, 13)]


def _split_frames(
    make_sequence_frame: Callable[..., pl.DataFrame],
    make_question_skill_frame: Callable[..., pl.DataFrame],
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    seq = make_sequence_frame(_USERS, _QUESTIONS, _LABELS, _TIMESTAMPS)
    rel = {"question_skill": make_question_skill_frame(_QS_PAIRS)}
    return seq, rel


# --- _build_split_sequences: skill expansion ------------------------------------


class TestBuildSplitSequencesSkills:
    def test_skill_column_aligned_with_questions(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq, rel = _split_frames(make_sequence_frame, make_question_skill_frame)
        ds = make_data_source(sequence_data=seq, relation_data=rel, max_seq_len=2)
        out = ds._build_split_sequences(expand_skills=True)

        assert out.columns == [
            "user",
            "question",
            "label",
            "timestamp",
            "sequence_id",
            "skill",
            "seq_pos",
        ]
        assert out.schema == {
            "user": pl.Int32,
            "question": pl.Int32,
            "label": pl.Int8,
            "timestamp": pl.Int64,
            "sequence_id": pl.Int32,
            "skill": pl.Int32,
            "seq_pos": pl.Int64,
        }
        # max_seq_len=2: each user's 4 expanded rows split into two 2-row
        # sub-sequences; dense ids follow global (user, split_idx) order.
        # NOTE: pinned current behavior -- a multi-skill question straddling a
        # split boundary has its skills land in different sub-sequences.
        expected = [
            (0, 1, 10, 0, 0),
            (0, 2, 11, 0, 1),
            (0, 2, 12, 1, 0),
            (0, 3, 13, 1, 1),
            (1, 1, 10, 2, 0),
            (1, 2, 11, 2, 1),
            (1, 2, 12, 3, 0),
            (1, 3, 13, 3, 1),
            (2, 1, 10, 4, 0),
            (2, 2, 11, 4, 1),
            (2, 2, 12, 5, 0),
            (2, 3, 13, 5, 1),
        ]
        got = out.select("user", "question", "skill", "sequence_id", "seq_pos")
        assert got.rows() == expected

    def test_order_determinism_with_scrambled_timestamps(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # Input rows arrive out of chronological order; the __order stamp must
        # restore interaction-major order with ascending skills per interaction.
        seq = make_sequence_frame(
            users=[0, 0, 0], questions=[2, 1, 2], labels=[1, 0, 0], timestamps=[3, 1, 2]
        )
        rel = {"question_skill": make_question_skill_frame(_QS_PAIRS)}
        ds = make_data_source(sequence_data=seq, relation_data=rel, max_seq_len=10)
        out = ds._build_split_sequences(expand_skills=True)

        assert out.height == 5
        assert out["timestamp"].to_list() == [1, 2, 2, 3, 3]
        assert out["question"].to_list() == [1, 2, 2, 2, 2]
        assert out["skill"].to_list() == [10, 11, 12, 11, 12]
        assert out["seq_pos"].to_list() == [0, 1, 2, 3, 4]
        assert out["sequence_id"].unique().to_list() == [0]


# --- _build_split_sequences: batch boundaries -----------------------------------


class TestBuildSplitSequencesBatches:
    @pytest.mark.parametrize("expand_skills", [False, True], ids=["question", "skill"])
    @pytest.mark.parametrize("limit", [1, 2, 4])
    def test_small_batch_limit_matches_single_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
        expand_skills: bool,
        limit: int,
    ) -> None:
        seq, rel = _split_frames(make_sequence_frame, make_question_skill_frame)

        ds_whole = make_data_source(sequence_data=seq, relation_data=rel, max_seq_len=2)
        whole = ds_whole._build_split_sequences(expand_skills=expand_skills)

        monkeypatch.setattr(DataSource, "_SPLIT_BATCH_ROWS", limit)
        ds_batched = make_data_source(
            sequence_data=seq, relation_data=rel, max_seq_len=2
        )
        batched = ds_batched._build_split_sequences(expand_skills=expand_skills)

        assert_frame_equal(whole, batched)

    def test_sequence_ids_dense_across_batches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # limit=1 forces every user into its own (oversized) batch; the global
        # counter still assigns dense ids in (user, split_idx) order.
        seq, rel = _split_frames(make_sequence_frame, make_question_skill_frame)
        monkeypatch.setattr(DataSource, "_SPLIT_BATCH_ROWS", 1)
        ds = make_data_source(sequence_data=seq, relation_data=rel, max_seq_len=2)
        out = ds._build_split_sequences(expand_skills=True)

        assert out["sequence_id"].n_unique() == 6
        assert sorted(out["sequence_id"].unique().to_list()) == list(range(6))

    def test_empty_input_returns_schema_only_frame(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq, rel = _split_frames(make_sequence_frame, make_question_skill_frame)
        ds = make_data_source(
            sequence_data=seq.head(0), relation_data=rel, max_seq_len=2
        )
        out = ds._build_split_sequences(expand_skills=True)

        assert out.height == 0
        assert out.columns == [
            "user",
            "question",
            "label",
            "timestamp",
            "sequence_id",
            "skill",
            "seq_pos",
        ]
        assert out.schema["sequence_id"] == pl.Int32
        assert out.schema["skill"] == pl.Int32
        assert out.schema["seq_pos"] == pl.Int64

    def test_empty_input_question_mode_omits_skill_column(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(_USERS, _QUESTIONS, _LABELS, _TIMESTAMPS)
        ds = make_data_source(sequence_data=seq.head(0), max_seq_len=2)
        out = ds._build_split_sequences(expand_skills=False)

        assert out.height == 0
        assert out.columns == [
            "user",
            "question",
            "label",
            "timestamp",
            "sequence_id",
            "seq_pos",
        ]

    def test_skill_expansion_without_relation_raises(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(_USERS, _QUESTIONS, _LABELS, _TIMESTAMPS)
        ds = make_data_source(sequence_data=seq, max_seq_len=2)
        with pytest.raises(ValueError, match="question_skill relation not available"):
            ds._build_split_sequences(expand_skills=True)


# --- build_split_* / build_windowlate guards ------------------------------------


class TestBuildGuards:
    def test_build_split_question_no_data_raises(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        ds = make_data_source(sequence_data=None)
        with pytest.raises(ValueError, match="No processed data available"):
            ds.build_split_question_sequence_data()

    def test_build_split_skill_no_data_raises(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        ds = make_data_source(sequence_data=None)
        with pytest.raises(ValueError, match="No processed data available"):
            ds.build_split_skill_sequence_data()

    def test_build_split_skill_missing_relation_raises(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame([0], [1])
        ds = make_data_source(sequence_data=seq, relation_data={})
        with pytest.raises(ValueError, match="question_skill relation not available"):
            ds.build_split_skill_sequence_data()

    def test_build_windowlate_missing_fold_column_raises(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame([0, 0], [1, 2])
        rel = {"question_skill": make_question_skill_frame(_QS_PAIRS)}
        ds = make_data_source(sequence_data=seq, relation_data=rel)
        with pytest.raises(ValueError, match="K-fold labels not found in data"):
            ds.build_windowlate_data()

    def test_build_windowlate_empty_test_fold_raises(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame([0, 0], [1, 2]).with_columns(
            pl.lit(0, dtype=pl.Int32).alias("fold")
        )
        rel = {"question_skill": make_question_skill_frame(_QS_PAIRS)}
        ds = make_data_source(sequence_data=seq, relation_data=rel)
        with pytest.raises(ValueError, match="No test-set interactions"):
            ds.build_windowlate_data()


# --- add_kfold_labels -------------------------------------------------------------


class TestAddKfoldLabels:
    @pytest.mark.parametrize("bad_ratio", [1.5, -0.1], ids=["above", "below"])
    def test_test_ratio_out_of_bounds_raises(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        bad_ratio: float,
    ) -> None:
        seq = make_sequence_frame([0, 0], [1, 2])
        ds = make_data_source(sequence_data=seq)
        with pytest.raises(ValueError, match="Test ratio should within 0~1"):
            ds.add_kfold_labels(test_ratio=bad_ratio)

    def test_user_level_fold_isolation(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # 10 users x 2 rows; test_ratio=0.2 -> int(10*0.2)=2 test users.
        users = [u for u in range(10) for _ in range(2)]
        seq = make_sequence_frame(users, list(range(20)))
        ds = make_data_source(sequence_data=seq)

        ds.add_kfold_labels(n_splits=4, test_ratio=0.2)

        out = ds.sequence_data
        assert "fold" in out.columns
        # Every user lands in exactly one fold.
        per_user = out.group_by("user").agg(pl.col("fold").n_unique().alias("n"))
        assert (per_user["n"] == 1).all()
        # Exactly 2 test users, all labeled -1.
        user_folds = out.select("user", "fold").unique()
        test_users = user_folds.filter(pl.col("fold") == -1)["user"].to_list()
        assert len(test_users) == 2
        # Non-test users cover every fold 0..3 and nothing else.
        train_folds = user_folds.filter(pl.col("fold") != -1)["fold"].to_list()
        assert sorted(train_folds) == [0, 0, 1, 1, 2, 2, 3, 3]
        assert ds.metadata["kfold_n_splits"] == 4
        assert ds.metadata["test_ratio"] == 0.2

    def test_same_seed_same_split(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # add_kfold_labels seeds its shuffle from the data source's seed, so
        # two sources with the same seed and data produce identical labels
        # without touching the global numpy RNG.
        users = [u for u in range(8) for _ in range(3)]
        seq = make_sequence_frame(users, list(range(24)))

        ds_a = make_data_source(sequence_data=seq, seed=7)
        ds_a.add_kfold_labels(n_splits=3, test_ratio=0.25)

        ds_b = make_data_source(sequence_data=seq, seed=7)
        ds_b.add_kfold_labels(n_splits=3, test_ratio=0.25)

        assert_frame_equal(ds_a.sequence_data, ds_b.sequence_data)

    def test_different_seed_different_split(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        users = [u for u in range(8) for _ in range(3)]
        seq = make_sequence_frame(users, list(range(24)))

        ds_a = make_data_source(sequence_data=seq, seed=7)
        ds_a.add_kfold_labels(n_splits=3, test_ratio=0.25)
        ds_b = make_data_source(sequence_data=seq, seed=8)
        ds_b.add_kfold_labels(n_splits=3, test_ratio=0.25)

        test_users_a = set(
            ds_a.sequence_data.filter(pl.col("fold") == -1)["user"].to_list()
        )
        test_users_b = set(
            ds_b.sequence_data.filter(pl.col("fold") == -1)["user"].to_list()
        )
        assert test_users_a != test_users_b

    def test_zero_test_ratio_folds_all_users(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        users = [u for u in range(6) for _ in range(2)]
        seq = make_sequence_frame(users, list(range(12)))
        ds = make_data_source(sequence_data=seq)

        ds.add_kfold_labels(n_splits=3, test_ratio=0.0)

        folds = ds.sequence_data["fold"].to_list()
        assert -1 not in folds
        assert sorted(set(folds)) == [0, 1, 2]
