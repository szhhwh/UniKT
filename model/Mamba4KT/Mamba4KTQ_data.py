"""Mamba4KTQ: question-level Mamba4KT data for the skill-vs-question ablation."""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class Mamba4KTQDataset(Dataset):
    """Question-level Mamba4KT sample: ``(question, response, mask)``."""

    def __init__(self, questions, responses, masks):
        self.questions = torch.from_numpy(questions).long()
        self.responses = torch.from_numpy(responses).long()
        self.masks = torch.from_numpy(masks).bool()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        return self.questions[idx], self.responses[idx], self.masks[idx]


class Mamba4KTQModelData(QuestionModelData):
    """Question-level data for the Mamba4KT ablation.

    The question id is the concept embedding unit (no Rasch pid), to contrast
    with the skill-level Mamba4KT. Test uses the standard K-fold split instead
    of windowlate, which is specific to KC (skill-level) models.
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

        dataset = Mamba4KTQDataset(user_question, user_response, user_mask)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"Mamba4KTQ data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset
