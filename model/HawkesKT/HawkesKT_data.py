from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class HawkesKTDataset(Dataset):
    def __init__(self, skill_seqs, problem_seqs, time_seqs, labels, masks):
        self.skill_seqs = skill_seqs
        self.problem_seqs = problem_seqs
        self.time_seqs = time_seqs
        self.labels = labels
        self.masks = masks

    def __len__(self) -> int:
        return len(self.skill_seqs)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.skill_seqs[idx], dtype=torch.long),
            torch.tensor(self.problem_seqs[idx], dtype=torch.long),
            torch.tensor(self.time_seqs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )


class HawkesKTModelData(QuestionModelData):
    """HawkesKT model data loader.

    Builds all sequences (skill, problem, time, label) from the question
    split data, selecting the first skill annotated on each interaction.
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        # 1. Load question split data (one row per question, has timestamp)
        sequences = self.load_sequences(
            required=("timestamp",), optional=("ms_first_response",)
        )
        partition = sequences.partition(rc.data.fold)
        skill_seqs = to_first_skill_ids(sequences.skills)
        problem_seqs = sequences.question
        label_seqs = sequences.response
        mask_seqs = sequences.mask
        if (
            self.data_src.dataset == "assistments09"
            and "ms_first_response" in sequences.features
        ):
            dwell = (
                np.nan_to_num(
                    sequences.features["ms_first_response"].astype(np.float64), nan=0.0
                )
                / 1000.0
            )
            increments = np.where(sequences.mask, dwell + 1.0, 0.0)
            times = np.cumsum(increments, axis=1) - increments
        else:
            times = sequences.features["timestamp"].astype(np.float64) / 1000.0
        time_seqs = np.where(sequences.mask, times, 0).astype(np.int64)

        # 4. K-fold split
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")

        dataset = HawkesKTDataset(
            skill_seqs, problem_seqs, time_seqs, label_seqs, mask_seqs
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.debug(
            f"HawkesKT data ready: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
