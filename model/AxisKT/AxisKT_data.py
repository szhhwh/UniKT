"""Question-level data preparation for AxisKT."""

import math
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class AxisKTDataset(Dataset):
    """Question sequences with a once-precomputed KC packing order."""

    def __init__(
        self,
        questions,
        responses,
        times,
        masks,
        question_skill_ids,
        question_skill_mask,
    ):
        self.questions = torch.from_numpy(np.asarray(questions)).long()
        self.responses = torch.from_numpy(np.asarray(responses)).long()
        self.times = torch.from_numpy(np.asarray(times)).double()
        self.masks = torch.from_numpy(np.asarray(masks)).bool()

        question_skill_ids = torch.from_numpy(np.asarray(question_skill_ids)).long()
        question_skill_mask = torch.from_numpy(np.asarray(question_skill_mask)).bool()
        skill_ids = question_skill_ids[self.questions]
        occurrence_mask = question_skill_mask[self.questions] & self.masks.unsqueeze(-1)
        _, seq_len, _ = skill_ids.shape
        positions = torch.arange(seq_len).view(1, seq_len, 1).expand_as(skill_ids)
        flat_skill = skill_ids.flatten(1)
        flat_position = positions.flatten(1)
        flat_valid = occurrence_mask.flatten(1)
        invalid_key = (int(question_skill_ids.max()) + 1) * (seq_len + 1)
        sort_key = flat_skill * (seq_len + 1) + flat_position
        sort_key = torch.where(
            flat_valid,
            sort_key,
            invalid_key + flat_position,
        )
        self.kc_order = torch.argsort(sort_key, dim=1, stable=True)
        self.kc_valid_counts = flat_valid.sum(dim=1)
        # Inverse permutation of kc_order over the full flat slot domain.
        kc_inverse = torch.empty_like(self.kc_order)
        kc_inverse.scatter_(
            1,
            self.kc_order,
            torch.arange(self.kc_order.size(1)).expand_as(self.kc_order),
        )
        self.kc_inverse = kc_inverse

    def __getitem__(self, index):
        return (
            self.questions[index],
            self.responses[index],
            self.times[index],
            self.masks[index],
            self.kc_order[index],
            self.kc_inverse[index],
            self.kc_valid_counts[index],
        )

    def __len__(self):
        return len(self.questions)


def axiskt_packed_collate_fn(batch):
    """Stack dense inputs and trim precomputed orders to the batch width.

    The returned 7-tuple appends ``valid_idx``: row-major indices of the
    adjacent-pair mask ``mask[:, :-1] & mask[:, 1:]`` flattened over
    ``[B, S-1]``. ``kc_inverse`` keeps the full flat slot width — the fused
    readout indexes it by (position, slot).
    """
    dense_columns = [torch.stack(column) for column in zip(*(row[:4] for row in batch))]
    packed_length = max(int(row[6]) for row in batch)
    kc_order = torch.stack([row[4][:packed_length] for row in batch])
    kc_inverse = torch.stack([row[5] for row in batch])
    masks = dense_columns[3]
    valid_mask = masks[:, :-1] & masks[:, 1:]
    valid_idx = valid_mask.flatten().nonzero().flatten()
    return (*dense_columns, kc_order, kc_inverse, valid_idx)


def use_compact_global(mask: torch.Tensor) -> bool:
    """Choose compact global computation from a batch mask.

    Only trailing padding can be skipped. An interior masked position remains
    active because later valid positions may depend on its encoded state.
    """
    if mask.numel() == 0:
        return False
    positions = torch.arange(1, mask.size(1) + 1, device=mask.device)
    lengths = torch.where(mask, positions, 0).amax(dim=1)
    active_count = int(lengths.sum())
    return active_count > 0 and 2 * active_count <= mask.numel()


def build_question_skill_table(data_src: DataSource) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded, permutation-stable KC set for every question."""
    relation = data_src.get_relation("question_skill")
    if isinstance(relation, pl.LazyFrame):
        relation = relation.collect()

    grouped = relation.group_by("question").agg(
        pl.col("skill").unique().sort().alias("skills")
    )
    num_questions = int(data_src.get_metadata("num_questions"))
    num_skills = int(data_src.get_metadata("num_skills"))
    max_skills = max(1, int(grouped.select(pl.col("skills").list.len().max()).item()))

    skill_ids = np.full((num_questions, max_skills), num_skills, dtype=np.int64)
    skill_mask = np.zeros((num_questions, max_skills), dtype=np.bool_)
    for row in grouped.iter_rows(named=True):
        question = int(row["question"])
        skills = np.asarray(row["skills"], dtype=np.int64)
        if question < 0 or question >= num_questions:
            raise ValueError(f"Question id {question} is outside metadata range")
        if skills.size and (skills.min() < 0 or skills.max() >= num_skills):
            raise ValueError(f"Question {question} has an out-of-range skill id")
        skill_ids[question, : skills.size] = skills
        skill_mask[question, : skills.size] = True

    missing = np.flatnonzero(~skill_mask.any(axis=1))
    if missing.size:
        logger.warning("AxisKT found %d questions without a KC relation", missing.size)
    logger.info("AxisKT question-KC table: max_skills_per_question=%d", max_skills)
    return skill_ids, skill_mask


def derive_max_gap_bins(time_seqs: np.ndarray) -> int:
    """Derive the log2-gap bucket count covering the largest time span."""
    max_span = max(1.0, float(time_seqs.max()))
    return max(2, int(math.floor(math.log2(max_span))) + 2)


def build_axiskt_model(rc, data_src: DataSource, extra: dict):
    """Build the AxisKT module from a run config and prepared extras."""
    from model.AxisKT.AxisKT_model import AxisKT

    m = rc.model
    return AxisKT(
        data_metadata=data_src.get_metadata(),
        question_skill_ids=extra["question_skill_ids"],
        question_skill_mask=extra["question_skill_mask"],
        hidden_dim=m.hidden_dim,
        n_blocks=m.n_blocks,
        max_gap_bins=int(extra["max_gap_bins"]),
        dropout=m.dropout,
        conv_kernel_size=m.conv_kernel_size,
        conv_dilation_base=m.conv_dilation_base,
        question_embed_dim=None if m.question_embed_dim < 0 else m.question_embed_dim,
        use_global=m.use_global,
        use_local=m.use_local,
        use_forgetting=getattr(m, "use_forgetting", True),
        use_rasch_embedding=m.use_rasch_embedding,
        use_irt_head=m.use_irt_head,
    )


class AxisKTModelData(QuestionModelData):
    """Prepare original question sequences and a separate question-KC view."""

    @override
    def prepare_data(self, rc: Any):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        if fold_idx is None:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        if fold_idx >= kfold_n_splits:
            raise ValueError(
                f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
            )

        questions, responses, masks, user_id_sequence = self.load_sequence_data()
        times = self._build_time_sequences()
        # Rows are subsequences, and user_id_sequence is zero-padded beyond
        # the mask; each row's first valid position carries its user id.
        user_ids = user_id_sequence[np.arange(len(masks)), masks.argmax(axis=1)]
        train_data, val_data, test_data = self.split_kfold_data(
            questions, responses, times, masks, user_ids, fold_idx=fold_idx
        )
        question_skill_ids, question_skill_mask = build_question_skill_table(
            self.data_src
        )
        max_gap_bins = derive_max_gap_bins(times)
        logger.info(
            "Derived max_gap_bins=%d from the largest intra-sequence time span",
            max_gap_bins,
        )
        logger.info("Using K-fold: fold %d/%d", fold_idx + 1, kfold_n_splits)
        return (
            AxisKTDataset(*train_data[:4], question_skill_ids, question_skill_mask),
            AxisKTDataset(*val_data[:4], question_skill_ids, question_skill_mask),
            AxisKTDataset(*test_data[:4], question_skill_ids, question_skill_mask),
            {
                "question_skill_ids": question_skill_ids,
                "question_skill_mask": question_skill_mask,
                "max_gap_bins": max_gap_bins,
                "user_ids": {
                    "train": train_data[4],
                    "val": val_data[4],
                    "test": test_data[4],
                },
            },
        )

    def _build_time_sequences(self) -> np.ndarray:
        """Return per-position interaction times in seconds (float64).

        A ``timestamp`` column (milliseconds) is used when present;
        ``assistments09`` stores ``order_id`` there instead, so its
        ``ms_first_response`` dwell times are accumulated. Sequences without
        a usable timestamp fall back to position indices.
        """
        q_data = self.load_split_data(optional=("ms_first_response", "timestamp"))
        num_users = q_data["sequence_id"].n_unique()
        max_seq_len = int(self.data_src.get_metadata("max_question_seq_len"))
        time_seqs = np.zeros((num_users, max_seq_len), dtype=np.float64)
        user_indices = q_data["sequence_id"].to_numpy()
        seq_positions = q_data["seq_pos"].to_numpy()
        columns = q_data.columns

        if self.data_src.dataset == "assistments09" and "ms_first_response" in columns:
            dwell = (
                q_data["ms_first_response"].fill_null(0).to_numpy().astype(np.float64)
                / 1000.0
            )
            cumshift = np.zeros(num_users, dtype=np.float64)
            for idx in np.lexsort((seq_positions, user_indices)):
                uid = user_indices[idx]
                pos = seq_positions[idx]
                time_seqs[uid, pos] = cumshift[uid]
                cumshift[uid] += dwell[idx] + 1.0
        elif "timestamp" in columns:
            time_seqs[user_indices, seq_positions] = (
                q_data["timestamp"].fill_null(0).to_numpy().astype(np.float64) / 1000.0
            )
        else:
            logger.warning(
                "No usable timestamp column for dataset %r; AxisKT time gaps "
                "fall back to position indices",
                self.data_src.dataset,
            )
            time_seqs[user_indices, seq_positions] = seq_positions.astype(np.float64)
        return time_seqs


__all__ = [
    "AxisKTDataset",
    "AxisKTModelData",
    "axiskt_packed_collate_fn",
    "build_axiskt_model",
    "build_question_skill_table",
    "derive_max_gap_bins",
    "use_compact_global",
]
