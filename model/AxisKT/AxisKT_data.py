"""Question-level data preparation for AxisKT."""

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import PackedSequences, to_multi_skill_ids

logger = get_logger(__name__)


class AxisKTDataset(Dataset):
    """Question sequences with a once-precomputed KC packing order."""

    def __init__(
        self,
        questions,
        responses,
        times,
        masks,
        skill_ids,
        skill_mask,
    ):
        self.questions = torch.from_numpy(np.asarray(questions)).long()
        self.responses = torch.from_numpy(np.asarray(responses)).long()
        self.times = torch.from_numpy(np.asarray(times)).double()
        self.masks = torch.from_numpy(np.asarray(masks)).bool()

        self.skill_ids = torch.from_numpy(np.asarray(skill_ids)).long()
        self.skill_mask = torch.from_numpy(np.asarray(skill_mask)).bool()
        skill_ids = self.skill_ids
        occurrence_mask = self.skill_mask & self.masks.unsqueeze(-1)
        _, seq_len, _ = skill_ids.shape
        positions = torch.arange(seq_len).view(1, seq_len, 1).expand_as(skill_ids)
        flat_skill = skill_ids.flatten(1)
        flat_position = positions.flatten(1)
        flat_valid = occurrence_mask.flatten(1)
        invalid_key = (int(skill_ids.max()) + 1) * (seq_len + 1)
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
            self.skill_ids[index],
            self.skill_mask[index],
        )

    def __len__(self):
        return len(self.questions)


def axiskt_packed_collate_fn(batch):
    """Stack dense inputs and trim precomputed orders to the batch width.

    ``valid_idx`` contains row-major indices of the
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
    skills = torch.stack([row[7] for row in batch])
    skill_mask = torch.stack([row[8] for row in batch])
    return (*dense_columns, kc_order, kc_inverse, valid_idx, skills, skill_mask)


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
        local_encoder=m.local_encoder,
    )


class AxisKTModelData(QuestionModelData):
    """Prepare question sequences with each event's own KC annotations."""

    @override
    def prepare_data(self, rc: Any):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        sequences = self.load_sequences(optional=("ms_first_response", "timestamp"))
        partition = sequences.partition(rc.data.fold)
        questions = sequences.question
        responses = sequences.response
        masks = sequences.mask
        times = self._build_time_sequences(sequences)
        user_ids = sequences.user
        skills = to_multi_skill_ids(
            sequences.skills, padding_value=self.data_src.get_metadata("num_skills")
        )
        skill_mask = sequences.skills >= 0
        dataset = AxisKTDataset(questions, responses, times, masks, skills, skill_mask)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)
        max_gap_bins = derive_max_gap_bins(times)
        logger.info(
            "Derived max_gap_bins=%d from the largest intra-sequence time span",
            max_gap_bins,
        )
        logger.info("Using K-fold: fold %d/%d", fold_idx + 1, kfold_n_splits)
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            {
                "max_gap_bins": max_gap_bins,
                "user_ids": {
                    "train": user_ids[partition.train],
                    "val": user_ids[partition.val],
                    "test": user_ids[partition.test],
                },
            },
        )

    def _build_time_sequences(self, sequences: PackedSequences) -> np.ndarray:
        """Return per-position interaction times in seconds (float64).

        A ``timestamp`` column (milliseconds) is used when present;
        ``assistments09`` stores ``order_id`` there instead, so its
        ``ms_first_response`` dwell times are accumulated. Sequences without
        a usable timestamp fall back to position indices.
        """
        features = sequences.features
        if self.data_src.dataset == "assistments09" and "ms_first_response" in features:
            dwell = (
                np.nan_to_num(features["ms_first_response"].astype(np.float64), nan=0.0)
                / 1000.0
            )
            increments = np.where(sequences.mask, dwell + 1.0, 0.0)
            time_seqs = np.cumsum(increments, axis=1) - increments
        elif "timestamp" in features:
            time_seqs = (
                np.nan_to_num(features["timestamp"].astype(np.float64), nan=0.0)
                / 1000.0
            )
        else:
            logger.warning(
                "No usable timestamp column for dataset %r; using sequence positions",
                self.data_src.dataset,
            )
            time_seqs = np.broadcast_to(
                np.arange(sequences.mask.shape[1], dtype=np.float64),
                sequences.mask.shape,
            )
        time_seqs = np.where(sequences.mask, time_seqs, 0.0)
        return time_seqs


__all__ = [
    "AxisKTDataset",
    "AxisKTModelData",
    "axiskt_packed_collate_fn",
    "build_axiskt_model",
    "derive_max_gap_bins",
    "use_compact_global",
]
