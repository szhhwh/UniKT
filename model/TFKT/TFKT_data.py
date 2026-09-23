"""TFKT 模型数据处理模块"""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class TFKTDataset(Dataset):
    """Training and validation sequences with question IDs for Rasch effects."""

    def __init__(self, sequences, responses, masks, questions):
        self.sequences = torch.from_numpy(sequences).long()
        self.responses = torch.from_numpy(responses).long()
        self.masks = torch.from_numpy(masks).bool()
        self.questions = torch.from_numpy(questions).long()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],
            self.responses[idx],
            self.masks[idx],
            self.questions[idx],
        )


class TFKTModelData(SkillModelData):
    """Build skill sequences and retain question IDs for the Rasch effect."""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """Return training, validation, and windowed test data."""
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        user_sequence, user_response, user_mask, _, user_question = (
            self.build_sequence_data()
        )

        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            train_data, val_data, _ = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
            train_question, val_question, _ = self.split_kfold_data(
                user_question, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        window_test_data = self.create_windowlate_iterable_dataset(rc.data.max_seq_len)

        train_dataset = TFKTDataset(
            train_data[0], train_data[1], train_data[2], train_question[0]
        )
        val_dataset = TFKTDataset(
            val_data[0], val_data[1], val_data[2], val_question[0]
        )
        pin_memory = rc.general.pin_memory
        if pin_memory is None:
            pin_memory = rc.general.device == "cuda" or (
                rc.general.device is None and torch.cuda.is_available()
            )
        test_dataset = DataLoader(
            window_test_data,
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        return train_dataset, val_dataset, test_dataset
