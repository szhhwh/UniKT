"""PSKT model data preparation."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_multi_skill_ids

logger = get_logger(__name__)


class PSKTDataset(Dataset):
    """PSKT dataset.

    Args:
        sequences: Question id sequence [N, S] (1-based, 0 = padding).
        responses: Response sequence [N, S] (0/1 valid, 2 = padding).
        masks: Valid position mask [N, S].
        skills: Multi-concept skill sequence [N, S, max_concepts] (1-based, -1 = empty).
        timestamps: Timestamp sequence [N, S].
    """

    def __init__(self, sequences, responses, masks, skills, timestamps):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.skills = skills
        self.timestamps = timestamps

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.skills[idx], dtype=torch.long),
            torch.tensor(self.timestamps[idx], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class PSKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.max_concepts = None

    @override
    def prepare_data(self, rc):
        """Prepare train, validation, and test datasets.

        Args:
            rc: RunConfig instance.

        Returns:
            (train_dataset, val_dataset, test_dataset) tuple.
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        sequences = self.load_sequences(required=("timestamp",))
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = np.where(sequences.mask, sequences.response, 2)
        user_mask = sequences.mask
        user_timestamp = sequences.features["timestamp"].astype(np.int64)

        user_skills = to_multi_skill_ids(sequences.skills, padding_value=-1, offset=1)
        self.max_concepts = user_skills.shape[-1]

        pad = ~user_mask.astype(bool)
        user_sequence = np.where(pad, 0, user_sequence + 1)
        user_skills[pad] = -1

        logger.info(f"PSKT data prepared: max_concepts={self.max_concepts}")

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = PSKTDataset(
            user_sequence, user_response, user_mask, user_skills, user_timestamp
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"PSKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
