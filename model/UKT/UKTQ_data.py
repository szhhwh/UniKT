"""UKTQ: question-level UKT data for the skill-vs-question ablation."""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from model.UKT.UKT_data import build_ukt_response_aug
from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class UKTQDataset(Dataset):
    """Question-level UKT sample: ``(question, response, mask, response_aug)``."""

    def __init__(self, questions, responses, masks, response_aug):
        self.questions = torch.from_numpy(questions).long()
        self.responses = torch.from_numpy(responses).long()
        self.masks = torch.from_numpy(masks).bool()
        self.response_aug = torch.from_numpy(response_aug).long()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        return (
            self.questions[idx],
            self.responses[idx],
            self.masks[idx],
            self.response_aug[idx],
        )


class UKTQModelData(QuestionModelData):
    """Question-level data for the UKT ablation.

    The question id is the concept embedding unit (no Rasch pid), to contrast
    with the skill-level UKT. The contrastive ``response_aug`` is granularity-
    independent (it flips response labels, never ids) and is reused verbatim
    from UKT_data. Test uses the standard K-fold split instead of windowlate.
    """

    def __init__(self, data_src):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_question = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        user_response_aug = build_ukt_response_aug(user_response, user_mask)

        dataset = UKTQDataset(
            user_question, user_response, user_mask, user_response_aug
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"UKTQ data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset
