"""Aligned sequence features and explicit fold membership."""

from collections.abc import Sized
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import polars as pl
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class SequencePartition:
    """Row indices into one shared sequence dataset."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @classmethod
    def from_folds(
        cls, folds: np.ndarray, fold_idx: int, num_folds: int
    ) -> "SequencePartition":
        """Build sample indices from explicit fold labels."""
        if fold_idx < 0 or fold_idx >= num_folds:
            raise ValueError(f"fold_idx {fold_idx} out of range [0, {num_folds})")
        if not np.all((folds >= -1) & (folds < num_folds)):
            raise ValueError("Fold labels are outside the declared fold range")
        return cls(
            train=np.flatnonzero((folds != fold_idx) & (folds != -1)),
            val=np.flatnonzero(folds == fold_idx),
            test=np.flatnonzero(folds == -1),
        )

    def apply(
        self, dataset: Dataset[Any]
    ) -> tuple[Subset[Any], Subset[Any], Subset[Any]]:
        """Select samples without copying their backing features."""
        if len(cast(Sized, dataset)) != sum(
            len(indices) for indices in (self.train, self.val, self.test)
        ):
            raise ValueError("Dataset length does not match the sequence partition")
        return (
            Subset(dataset, self.train.tolist()),
            Subset(dataset, self.val.tolist()),
            Subset(dataset, self.test.tolist()),
        )


@dataclass(frozen=True)
class PackedSequences:
    """Features whose first dimension shares one explicit sequence order."""

    question: np.ndarray
    response: np.ndarray
    mask: np.ndarray
    skills: np.ndarray
    sequence_id: np.ndarray
    user: np.ndarray
    fold: np.ndarray
    features: dict[str, np.ndarray]
    num_folds: int

    def partition(self, fold_idx: int) -> SequencePartition:
        """Use one validation fold, held-out test users, and remaining training users."""
        return SequencePartition.from_folds(self.fold, fold_idx, self.num_folds)


def pack_sequences(
    data: pl.DataFrame, max_seq_len: int, num_folds: int
) -> PackedSequences:
    """Pack a projected frame once, retaining sequence identity and fold alignment."""
    if data.is_empty():
        raise ValueError("Cannot pack empty split sequences")
    membership = (
        data.group_by("sequence_id")
        .agg(
            pl.col("user").first(),
            pl.col("fold").first(),
            pl.col("user").n_unique().alias("num_users"),
            pl.col("fold").n_unique().alias("num_folds"),
        )
        .sort("sequence_id")
    )
    if membership.filter(
        (pl.col("num_users") != 1) | (pl.col("num_folds") != 1)
    ).height:
        raise ValueError("Each sequence must belong to one user and one fold")
    folds = membership["fold"].to_numpy()
    if membership["fold"].null_count() or np.any((folds < -1) | (folds >= num_folds)):
        raise ValueError("Sequence fold labels are outside the declared fold range")
    user_folds = membership.group_by("user").agg(pl.col("fold").n_unique())
    if user_folds.filter(pl.col("fold") != 1).height:
        raise ValueError("A student's sequences must belong to the same fold")

    sequence_ids = membership["sequence_id"].to_numpy()
    rows = np.searchsorted(sequence_ids, data["sequence_id"].to_numpy())
    positions = data["seq_pos"].to_numpy()
    if (
        data["seq_pos"].null_count()
        or np.any(positions < 0)
        or np.any(positions >= max_seq_len)
        or data.select(pl.struct("sequence_id", "seq_pos").is_duplicated().any()).item()
    ):
        raise ValueError(
            "Sequence positions must be unique and within the padded length"
        )
    shape = (len(sequence_ids), max_seq_len)

    def scatter(column: str, padding: int = 0) -> np.ndarray:
        values = data[column].to_numpy()
        result = np.full(shape, padding, dtype=values.dtype)
        result[rows, positions] = values
        return result

    mask = np.zeros(shape, dtype=np.bool_)
    mask[rows, positions] = True
    if "skills" in data.columns:
        width = cast(int, data["skills"].list.len().max())
        skills = np.full((*shape, width), -1, dtype=np.int64)
        for row, position, annotated in zip(
            rows, positions, data["skills"], strict=True
        ):
            skills[row, position, : len(annotated)] = annotated
    else:
        skills = scatter("skill", padding=-1).astype(np.int64)[..., None]

    reserved = {
        "question",
        "label",
        "skills",
        "skill",
        "sequence_id",
        "seq_pos",
        "user",
        "fold",
    }
    return PackedSequences(
        question=scatter("question").astype(np.int64),
        response=scatter("label").astype(np.int64),
        mask=mask,
        skills=skills,
        sequence_id=sequence_ids,
        user=membership["user"].to_numpy(),
        fold=folds,
        features={
            column: scatter(column, padding=-1 if column == "interaction_id" else 0)
            for column in data.columns
            if column not in reserved
        },
        num_folds=num_folds,
    )


def to_multi_skill_ids(
    skills: np.ndarray, padding_value: int = -1, offset: int = 0
) -> np.ndarray:
    """Return [N, S, K] skill IDs with the requested offset and padding."""
    return np.where(skills >= 0, skills + offset, padding_value)


def to_first_skill_ids(skills: np.ndarray, offset: int = 0) -> np.ndarray:
    """Return [N, S] first-annotated skill IDs, using zero for absent positions."""
    primary = skills[..., 0]
    return np.where(primary >= 0, primary + offset, 0)


def to_skill_combination_ids(skills: np.ndarray) -> tuple[np.ndarray, int]:
    """Return [N, S] ordered-combination IDs and their count; zero is padding."""
    valid = (skills >= 0).any(axis=-1)
    combinations = [tuple(row[row >= 0]) for row in skills[valid]]
    vocabulary = {
        value: index + 1 for index, value in enumerate(sorted(set(combinations)))
    }
    result = np.zeros(valid.shape, dtype=np.int64)
    result[valid] = [vocabulary[value] for value in combinations]
    return result, len(vocabulary)
