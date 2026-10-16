"""Tests for DataSource split-sequence building, build_* guards, add_kfold_labels."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

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
            "interaction_id",
            "sequence_id",
            "skill",
            "seq_pos",
        ]
        assert out.schema == {
            "user": pl.Int32,
            "question": pl.Int32,
            "label": pl.Int8,
            "timestamp": pl.Int64,
            "interaction_id": pl.Int64,
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


# --- _build_split_sequences: interaction-unit splitting --------------------------
#
# One user answers questions [1, 2, 3, 4] whose skills expand to
# [10], [11, 12, 13], [14], [15] -> 4 interactions, 6 expanded rows.
_IU_USERS = [0] * 4
_IU_QUESTIONS = [1, 2, 3, 4]
_IU_LABELS = [1, 0, 1, 1]
_IU_TIMESTAMPS = [1, 2, 3, 4]
_IU_QS_PAIRS = [(1, 10), (2, 11), (2, 12), (2, 13), (3, 14), (4, 15)]


class TestBuildSplitSequencesInteraction:
    def test_skill_unit_cuts_mid_interaction(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # max_seq_len=3 counts expanded rows: [10, 11, 12 | 13, 14, 15] --
        # question 2's skills straddle both splits.
        seq = make_sequence_frame(_IU_USERS, _IU_QUESTIONS, _IU_LABELS, _IU_TIMESTAMPS)
        rel = {"question_skill": make_question_skill_frame(_IU_QS_PAIRS)}
        ds = make_data_source(
            sequence_data=seq, relation_data=rel, max_seq_len=3, min_seq_len=2
        )
        out = ds._build_split_sequences(expand_skills=True)

        assert out["sequence_id"].to_list() == [0] * 3 + [1] * 3
        assert out["skill"].to_list() == [10, 11, 12, 13, 14, 15]
        assert out["question"].to_list() == [1, 2, 2, 2, 3, 4]
        assert out["seq_pos"].to_list() == [0, 1, 2, 0, 1, 2]

    def test_interaction_unit_keeps_interactions_whole(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # max_seq_len=3 counts interactions: [1, 2, 3 | 4]. The first split
        # expands to 5 rows (beyond max_seq_len); the 1-interaction tail is
        # dropped by min_seq_len=2, so question 4 leaves the dataset.
        seq = make_sequence_frame(_IU_USERS, _IU_QUESTIONS, _IU_LABELS, _IU_TIMESTAMPS)
        rel = {"question_skill": make_question_skill_frame(_IU_QS_PAIRS)}
        ds = make_data_source(
            sequence_data=seq,
            relation_data=rel,
            max_seq_len=3,
            min_seq_len=2,
            truncation_stage="pre_expansion",
        )
        out = ds._build_split_sequences(expand_skills=True)

        assert out["sequence_id"].to_list() == [0] * 5
        assert out["skill"].to_list() == [10, 11, 12, 13, 14]
        assert out["question"].to_list() == [1, 2, 2, 2, 3]
        # seq_pos is the split-internal expanded row index and exceeds
        # max_seq_len's budget without wrapping.
        assert out["seq_pos"].to_list() == [0, 1, 2, 3, 4]

    def test_interaction_unit_min_len_counts_interactions(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # Same data with min_seq_len=1: the 1-interaction tail survives as a
        # second split of a single row.
        seq = make_sequence_frame(_IU_USERS, _IU_QUESTIONS, _IU_LABELS, _IU_TIMESTAMPS)
        rel = {"question_skill": make_question_skill_frame(_IU_QS_PAIRS)}
        ds = make_data_source(
            sequence_data=seq,
            relation_data=rel,
            max_seq_len=3,
            min_seq_len=1,
            truncation_stage="pre_expansion",
        )
        out = ds._build_split_sequences(expand_skills=True)

        assert out["sequence_id"].to_list() == [0] * 5 + [1]
        assert out["skill"].to_list() == [10, 11, 12, 13, 14, 15]
        assert out["seq_pos"].to_list() == [0, 1, 2, 3, 4, 0]

    def test_question_mode_ignores_truncation_stage(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # The knob only governs the skill path: question sequences split
        # identically regardless of truncation_stage.
        seq = make_sequence_frame(_IU_USERS, _IU_QUESTIONS, _IU_LABELS, _IU_TIMESTAMPS)
        rel = {"question_skill": make_question_skill_frame(_IU_QS_PAIRS)}
        out_skill = make_data_source(
            sequence_data=seq, relation_data=rel, max_seq_len=2, min_seq_len=1
        )._build_split_sequences(expand_skills=False)
        out_interaction = make_data_source(
            sequence_data=seq,
            relation_data=rel,
            max_seq_len=2,
            min_seq_len=1,
            truncation_stage="pre_expansion",
        )._build_split_sequences(expand_skills=False)

        assert_frame_equal(out_skill, out_interaction)
        assert out_skill["sequence_id"].n_unique() == 2

    def test_schema_matches_skill_unit_output(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(_IU_USERS, _IU_QUESTIONS, _IU_LABELS, _IU_TIMESTAMPS)
        rel = {"question_skill": make_question_skill_frame(_IU_QS_PAIRS)}
        ds = make_data_source(
            sequence_data=seq,
            relation_data=rel,
            max_seq_len=3,
            min_seq_len=1,
            truncation_stage="pre_expansion",
        )
        out = ds._build_split_sequences(expand_skills=True)

        assert out.columns == [
            "user",
            "question",
            "label",
            "timestamp",
            "interaction_id",
            "sequence_id",
            "skill",
            "seq_pos",
        ]
        assert out.schema == {
            "user": pl.Int32,
            "question": pl.Int32,
            "label": pl.Int8,
            "timestamp": pl.Int64,
            "interaction_id": pl.Int64,
            "sequence_id": pl.Int32,
            "skill": pl.Int32,
            "seq_pos": pl.Int64,
        }


class TestBuildSplitSequencesUserColumn:
    """Split pipeline preserves the pre-split student id (user)."""

    def test_user_column_present_and_matches_source(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(
            [0, 0, 0, 1, 1], [1, 2, 3, 4, 5], [1, 0, 1, 0, 1], [1, 2, 3, 1, 2]
        )
        ds = make_data_source(sequence_data=seq, max_seq_len=2, min_seq_len=1)
        out = ds._build_split_sequences(expand_skills=False)

        assert "user" in out.columns
        assert "sequence_id" in out.columns
        # user 0 -> splits (0,0),(0,1) -> new ids 0,1; user 1 -> split id 2.
        # New ids are assigned per (user, split_idx) in global order.
        assert out["sequence_id"].to_list() == [0, 0, 1, 2, 2]
        assert out["user"].to_list() == [0, 0, 0, 1, 1]
        assert out["seq_pos"].to_list() == [0, 1, 0, 0, 1]

    def test_empty_input_keeps_schema(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame([], [], [], [])
        ds = make_data_source(sequence_data=seq, max_seq_len=2, min_seq_len=1)
        out = ds._build_split_sequences(expand_skills=False)

        assert "user" in out.columns
        assert "sequence_id" in out.columns
        assert "original_user" not in out.columns
        assert out.height == 0

    def test_min_seq_len_drop_keeps_global_id_order(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # user 0: 4 interactions -> two full splits (ids 0,1); user 1: 1
        # interaction -> dropped by min_seq_len=2 -> user 2's single split
        # takes dense id 2 (ids stay dense, ordered by (user, split_idx)).
        seq = make_sequence_frame(
            [0, 0, 0, 0, 1, 2, 2],
            [1, 2, 3, 4, 5, 6, 7],
            [1, 0, 1, 0, 1, 1, 0],
            [1, 2, 3, 4, 1, 1, 2],
        )
        ds = make_data_source(sequence_data=seq, max_seq_len=2, min_seq_len=2)
        out = ds._build_split_sequences(expand_skills=False)

        assert out["sequence_id"].to_list() == [0, 0, 1, 1, 2, 2]
        assert out["user"].to_list() == [0, 0, 0, 0, 2, 2]
        assert out["seq_pos"].to_list() == [0, 1, 0, 1, 0, 1]


class TestBuildSplitSequencesBatches:
    @pytest.mark.parametrize("expand_skills", [False, True], ids=["question", "skill"])
    @pytest.mark.parametrize("truncation_stage", ["post_expansion", "pre_expansion"])
    @pytest.mark.parametrize("limit", [1, 2, 4])
    def test_small_batch_limit_matches_single_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
        expand_skills: bool,
        truncation_stage: str,
        limit: int,
    ) -> None:
        seq, rel = _split_frames(make_sequence_frame, make_question_skill_frame)

        ds_whole = make_data_source(
            sequence_data=seq,
            relation_data=rel,
            max_seq_len=2,
            truncation_stage=truncation_stage,
        )
        whole = ds_whole._build_split_sequences(expand_skills=expand_skills)

        monkeypatch.setattr(DataSource, "_SPLIT_BATCH_ROWS", limit)
        ds_batched = make_data_source(
            sequence_data=seq,
            relation_data=rel,
            max_seq_len=2,
            truncation_stage=truncation_stage,
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
            "interaction_id",
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
            "interaction_id",
            "skills",
            "sequence_id",
            "seq_pos",
        ]

    def test_skill_expansion_uses_annotations_without_relation(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame(_USERS, _QUESTIONS, _LABELS, _TIMESTAMPS)
        ds = make_data_source(sequence_data=seq, max_seq_len=2)
        out = ds._build_split_sequences(expand_skills=True)
        assert out["skill"].to_list() == _QUESTIONS


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

    def test_build_split_skill_missing_annotations_raises(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame([0], [1])
        ds = make_data_source(sequence_data=seq, relation_data={})
        ds.sequence_data = ds.sequence_data.drop("skills")
        with pytest.raises(ValueError, match="interactions must provide skills"):
            ds.set_sequence_data(ds.sequence_data)

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


# --- save_data: padded-length metadata keys --------------------------------------


class TestSaveDataPaddedLengthMetadata:
    """max_question_seq_len / max_skill_seq_len are real per-split maxima."""

    USERS: ClassVar[list[int]] = [0] * 5 + [1] * 3
    QUESTIONS: ClassVar[list[int]] = [1, 2, 3, 1, 2, 1, 2, 3]
    QS_PAIRS: ClassVar[list[tuple[int, int]]] = [
        (1, 10),
        (2, 11),
        (3, 13),
    ]  # 1:1 -> skill rows == question rows

    def test_reprocessing_invalidates_previous_windowlate_artifact(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        tmp_path: Path,
    ) -> None:
        source = make_data_source(
            sequence_data=make_sequence_frame([0, 0], [0, 0]),
            metadata={
                "windowlate_data_md5": "obsolete",
                "max_windowlate_seq_len": 10,
            },
        )
        source.set_sequence_data(source.sequence_data)
        source.build_split_question_sequence_data()
        source.build_split_skill_sequence_data()
        (tmp_path / "stub_windowlate.parquet").write_bytes(b"obsolete")

        source.save_data()

        assert "windowlate_data_md5" not in source.metadata
        assert "max_windowlate_seq_len" not in source.metadata

    def _saved_metadata(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
        max_seq_len: int,
    ) -> dict:
        import json
        import os

        seq = make_sequence_frame(self.USERS, self.QUESTIONS)
        rel = {"question_skill": make_question_skill_frame(self.QS_PAIRS)}
        ds = make_data_source(
            sequence_data=seq, relation_data=rel, max_seq_len=max_seq_len
        )
        ds.split_question_sequence_data = ds._build_split_sequences(expand_skills=False)
        ds.split_skill_sequence_data = ds._build_split_sequences(expand_skills=True)
        ds.save_data()

        with open(os.path.join(ds.data_folder, "metadata.json")) as f:
            return json.load(f)

    def test_question_max_equals_budget_when_a_split_fills_it(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # max_seq_len=4: user 0 splits into [4, 1] rows, user 1 into [3] ->
        # the longest split fills the budget exactly.
        meta = self._saved_metadata(
            make_data_source,
            make_sequence_frame,
            make_question_skill_frame,
            max_seq_len=4,
        )
        assert meta["max_question_seq_len"] == 4

    def test_question_max_below_budget_when_all_splits_are_short(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        # max_seq_len=10 > every user: one split per user, longest is 5.
        meta = self._saved_metadata(
            make_data_source,
            make_sequence_frame,
            make_question_skill_frame,
            max_seq_len=10,
        )
        assert meta["max_question_seq_len"] == 5
        assert meta["max_question_seq_len"] < 10

    def test_bare_max_seq_len_key_is_gone(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        meta = self._saved_metadata(
            make_data_source,
            make_sequence_frame,
            make_question_skill_frame,
            max_seq_len=4,
        )
        assert "max_seq_len" not in meta
        # 1:1 question->skill mapping, so both real maxima coincide.
        assert meta["max_skill_seq_len"] == meta["max_question_seq_len"]

    def test_no_split_sequences_stops_before_writing(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
        tmp_path: Path,
    ) -> None:
        seq = make_sequence_frame([0, 0], [1, 2])
        rel = {"question_skill": make_question_skill_frame([(1, 10), (2, 11)])}
        ds = make_data_source(sequence_data=seq, relation_data=rel, min_seq_len=3)
        ds.split_question_sequence_data = ds._build_split_sequences(expand_skills=False)
        ds.split_skill_sequence_data = ds._build_split_sequences(expand_skills=True)

        with pytest.raises(ValueError, match="No question or skill split sequences"):
            ds.save_data()

        assert list(tmp_path.iterdir()) == []

    def test_one_empty_split_family_records_zero(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        make_question_skill_frame: Callable[..., pl.DataFrame],
    ) -> None:
        seq = make_sequence_frame([0, 0], [1, 2])
        rel = {
            "question_skill": make_question_skill_frame(
                [(1, 10), (1, 11), (2, 12), (2, 13)]
            )
        }
        ds = make_data_source(sequence_data=seq, relation_data=rel, min_seq_len=3)
        ds.split_question_sequence_data = ds._build_split_sequences(expand_skills=False)
        ds.split_skill_sequence_data = ds._build_split_sequences(expand_skills=True)

        ds.save_data()

        assert ds.get_metadata("max_question_seq_len") == 0
        assert ds.get_metadata("max_skill_seq_len") == 4
