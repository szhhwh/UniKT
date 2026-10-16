import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import TensorDataset

from utils.model_data.packed_sequences import pack_sequences


@pytest.fixture
def events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sequence_id": [40, 40, 10, 10, 90, 70],
            "seq_pos": [0, 1, 0, 1, 0, 0],
            "user": [12, 12, 7, 7, 12, 19],
            "fold": [1, 1, 0, 0, 1, -1],
            "question": [3, 3, 2, 1, 4, 5],
            "label": [1, 0, 0, 1, 1, 0],
            "skills": [[0], [1, 2], [2], [1], [0], [2]],
            "timestamp": [100, 200, 10, 20, 300, 90],
            "interaction_id": [4, 5, 1, 2, 6, 3],
        }
    )


def test_sparse_sequence_identity_and_features_stay_aligned(
    events: pl.DataFrame,
) -> None:
    packed = pack_sequences(events.reverse(), max_seq_len=3, num_folds=2)
    assert packed.sequence_id.tolist() == [10, 40, 70, 90]
    assert packed.user.tolist() == [7, 12, 19, 12]
    assert packed.fold.tolist() == [0, 1, -1, 1]
    assert packed.question.tolist() == [[2, 1, 0], [3, 3, 0], [5, 0, 0], [4, 0, 0]]
    assert packed.skills[1].tolist() == [[0, -1], [1, 2], [-1, -1]]
    assert packed.features["timestamp"][1].tolist() == [100, 200, 0]
    assert packed.features["interaction_id"][1].tolist() == [4, 5, -1]
    assert packed.mask.dtype == np.bool_
    assert packed.mask.sum() == len(events)


def test_partition_preserves_student_isolation_and_shares_backing_storage(
    events: pl.DataFrame,
) -> None:
    packed = pack_sequences(events, 3, 2)
    partition = packed.partition(0)
    assert partition.train.tolist() == [1, 3]
    assert partition.val.tolist() == [0]
    assert partition.test.tolist() == [2]
    tensor = torch.from_numpy(packed.question)
    dataset = TensorDataset(tensor)
    train, val, test = partition.apply(dataset)
    assert train.dataset is val.dataset is test.dataset is dataset
    assert len(train) + len(val) + len(test) == len(dataset)
    for indices, view in zip(
        (partition.train, partition.val, partition.test),
        (train, val, test),
        strict=True,
    ):
        for index, row in enumerate(indices):
            assert view[index][0].data_ptr() == tensor[row].data_ptr()
    tensor[1, 0] = 99
    assert train[0][0][0] == 99


def test_skill_chain_uses_singleton_annotations_and_same_partition(
    events: pl.DataFrame,
) -> None:
    scalar = events.drop("skills").with_columns(pl.lit(2).alias("skill"))
    packed = pack_sequences(scalar, 3, 2)
    assert packed.skills.shape == (4, 3, 1)
    assert (packed.skills[..., 0][packed.mask] == 2).all()
    assert (packed.skills[..., 0][~packed.mask] == -1).all()
    assert packed.partition(0).train.tolist() == [1, 3]


@pytest.mark.parametrize("fold", [-1, 2])
def test_invalid_requested_fold_fails(events: pl.DataFrame, fold: int) -> None:
    with pytest.raises(ValueError, match="fold_idx"):
        pack_sequences(events, 3, 2).partition(fold)


def test_partition_rejects_dataset_of_different_length(events: pl.DataFrame) -> None:
    partition = pack_sequences(events, 3, 2).partition(0)
    with pytest.raises(ValueError, match="Dataset length"):
        partition.apply(TensorDataset(torch.zeros(3, 2)))


def test_repeated_sequence_must_have_one_student(events: pl.DataFrame) -> None:
    events = events.with_columns(
        pl.when(pl.col("interaction_id") == 5)
        .then(99)
        .otherwise(pl.col("user"))
        .alias("user")
    )
    with pytest.raises(ValueError, match="one user and one fold"):
        pack_sequences(events, 3, 2)


def test_students_cannot_cross_folds_through_different_sequences(
    events: pl.DataFrame,
) -> None:
    events = events.with_columns(
        pl.when(pl.col("sequence_id") == 90)
        .then(0)
        .otherwise(pl.col("fold"))
        .alias("fold")
    )
    with pytest.raises(ValueError, match="same fold"):
        pack_sequences(events, 3, 2)


@pytest.mark.parametrize("position", [-1, 3])
def test_position_must_fit_padded_length(events: pl.DataFrame, position: int) -> None:
    events = events.with_columns(
        pl.when(pl.col("interaction_id") == 5)
        .then(position)
        .otherwise(pl.col("seq_pos"))
        .alias("seq_pos")
    )
    with pytest.raises(ValueError, match="Sequence positions"):
        pack_sequences(events, 3, 2)


def test_duplicate_positions_fail_at_packing(events: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="Sequence positions"):
        pack_sequences(pl.concat([events, events.head(1)]), 3, 2)


def test_empty_sequences_fail_at_packing(events: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty split"):
        pack_sequences(events.head(0), 3, 2)
