"""Data adapter for MCKT."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class MCKTDataset(Dataset):
    """Left-padded question sequence dataset matching the reference MCKT code."""

    def __init__(self, questions, responses, masks):
        self.questions = questions
        self.responses = responses
        self.masks = masks

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )


class MCKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_questions = sequences.question
        user_responses = sequences.response
        user_masks = sequences.mask
        user_questions, user_responses, user_masks = self._left_pad_sequences(
            user_questions, user_responses, user_masks
        )

        pos_matrix = self.build_pos_matrix(
            strategy=rc.model.pos_strategy,
            include_self=rc.model.pos_include_self,
        )

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = MCKTDataset(user_questions, user_responses, user_masks)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"MCKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test={len(test_dataset)}, pos_matrix={tuple(pos_matrix.shape)}"
        )
        return train_dataset, val_dataset, test_dataset, pos_matrix

    def build_pos_matrix(
        self,
        strategy: str = "shared_kc",
        include_self: bool = True,
    ) -> torch.Tensor:
        """Build question-question positive matrix from question-KC relations.

        Input: question_skill relation, converted to Q-matrix with shape
        [num_questions, num_skills].
        Output: float tensor with shape [num_questions, num_questions], where 1
        marks a positive question pair for MCKT question/interaction CL.
        """
        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        q_bool = q_matrix.astype(bool)

        if strategy == "same_kc_set":
            pos = (q_bool[:, None, :] == q_bool[None, :, :]).all(axis=-1)
        elif strategy == "shared_kc":
            pos = (q_bool.astype(np.int32) @ q_bool.astype(np.int32).T) > 0
        else:
            raise ValueError(
                f"Unsupported pos_strategy={strategy!r}. "
                "Supported: 'same_kc_set', 'shared_kc'."
            )

        if include_self:
            np.fill_diagonal(pos, True)
        else:
            np.fill_diagonal(pos, False)

        empty_rows = np.where(pos.sum(axis=1) == 0)[0]
        if len(empty_rows) > 0:
            logger.warning(
                f"pos_matrix has {len(empty_rows)} rows without positives; "
                "falling back to self-positive for those rows."
            )
            pos[empty_rows, empty_rows] = True

        return torch.from_numpy(pos.astype(np.float32))

    @staticmethod
    def _left_pad_sequences(questions, responses, masks):
        questions_out = np.zeros_like(questions)
        responses_out = np.full_like(responses, -1)
        masks_out = np.zeros_like(masks, dtype=bool)

        for idx in range(questions.shape[0]):
            valid = masks[idx].astype(bool)
            length = int(valid.sum())
            if length == 0:
                continue
            questions_out[idx, -length:] = questions[idx, valid]
            responses_out[idx, -length:] = responses[idx, valid]
            masks_out[idx, -length:] = True

        return questions_out, responses_out, masks_out
