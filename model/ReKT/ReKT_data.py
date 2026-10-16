from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_skill_combination_ids

logger = get_logger(__name__)


class ReKTDataset(Dataset):
    def __init__(self, questions, skills, responses, masks):
        self.questions = questions
        self.skills = skills
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.questions[index], dtype=torch.long),
            torch.tensor(self.skills[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.questions)


class ReKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask

        user_skill_sequence, num_combined_skills = to_skill_combination_ids(
            sequences.skills
        )
        user_skill_sequence = np.maximum(user_skill_sequence - 1, 0)

        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = ReKTDataset(
            user_sequence, user_skill_sequence, user_response, user_mask
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            {"num_combined_skills": num_combined_skills},
        )
