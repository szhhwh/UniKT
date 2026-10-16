"""SAKT data adapter."""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class SAKTDataset(Dataset):
    """Skill-level sequence dataset for SAKT.

    Training/validation returns:
        (sequence, response, mask)

    Windowlate evaluation is provided by SkillModelData's iterable dataset and
    is handled in SAKTTrainer.test_forward_pass.
    """

    def __init__(self, sequences, responses, masks):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        return sequence, response, mask


class SAKTModelData(SkillModelData):
    """Prepare skill-level sequences for SAKT."""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_sequence = to_first_skill_ids(sequences.skills)
        user_response = sequences.response
        user_mask = sequences.mask

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = SAKTDataset(user_sequence, user_response, user_mask)
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"SAKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
