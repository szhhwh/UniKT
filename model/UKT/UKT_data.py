"""UKT 模型数据处理模块"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class UKTDataset(Dataset):
    """UKT 数据集

    Args:
        sequences: 技能ID序列
        responses: 响应序列
        masks: 掩码序列
        questions: 题目ID序列
        response_aug: 对比学习增强响应序列
        late_group_ids: 题目级分组ID
        true_labels: 真实标签序列
    """

    def __init__(
        self,
        sequences,
        responses,
        masks,
        questions=None,
        response_aug=None,
        late_group_ids=None,
        true_labels=None,
    ):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions
        self.response_aug = response_aug
        self.late_group_ids = late_group_ids
        self.true_labels = true_labels
        self._is_window_mode = late_group_ids is not None and true_labels is not None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        question = (
            torch.tensor(self.questions[idx], dtype=torch.long)
            if self.questions is not None
            else None
        )
        response_aug = (
            torch.tensor(self.response_aug[idx], dtype=torch.long)
            if self.response_aug is not None
            else None
        )

        if self._is_window_mode:
            late_group_id = torch.tensor(self.late_group_ids[idx], dtype=torch.long)
            true_labels = torch.tensor(self.true_labels[idx], dtype=torch.long)
            if question is not None:
                return sequence, response, mask, late_group_id, true_labels, question
            return sequence, response, mask, late_group_id, true_labels

        if question is not None:
            if response_aug is not None:
                return sequence, response, mask, question, response_aug
            return sequence, response, mask, question
        return sequence, response, mask


def build_ukt_response_aug(responses, masks):
    """Build UKT contrastive learning augmented responses.

    Augmentation algorithm:
    - r_aug[0] flipped if rshft_aug_mask[-1] == r_aug_mask[0]
    - rshft_aug[0] flipped if rshft_aug[0] == rshft_aug_mask[-1]

    Note: the original's loop has a self-referencing bug where rshft_aug[0]
    is read on every iteration but gets modified on i=0, so only position 0
    of rshft_aug (= position 1 of target_aug) is ever flipped.
    """
    responses = np.asarray(responses)
    masks = np.asarray(masks).astype(bool)
    aug = responses.copy()

    if responses.ndim != 2 or responses.shape[1] < 2:
        return aug

    # Pair-validity: both adjacent positions must be valid (matches original mseqs)
    adjacent_mask = masks[:, :-1] & masks[:, 1:]
    has_pairs = adjacent_mask.any(axis=1)
    if not has_pairs.any():
        return aug

    first_idx = np.argmax(adjacent_mask, axis=1)
    last_idx = adjacent_mask.shape[1] - 1 - np.argmax(adjacent_mask[:, ::-1], axis=1)
    batch_idx = np.arange(responses.shape[0])

    last_valid_shft = responses[batch_idx, last_idx + 1]
    first_valid_r = responses[batch_idx, first_idx]

    # rshft_aug[0] = response[1] * mseqs[0] (zeroed if first pair invalid)
    rshft_aug_0 = responses[:, 1] * adjacent_mask[:, 0]

    # Condition 1: flip position 0 (r_aug[0] in original)
    flip_pos0 = has_pairs & (last_valid_shft == first_valid_r)
    aug[flip_pos0, 0] = 1 - aug[flip_pos0, 0]

    # Condition 2: flip position 1 (rshft_aug[0] in original)
    # Original checks rshft_aug[0] == rshft_aug_mask[-1],
    # with rshft_aug[0] = response[1] * mseqs[0]
    flip_pos1 = has_pairs & (rshft_aug_0 == last_valid_shft)
    aug[flip_pos1, 1] = 1 - aug[flip_pos1, 1]
    return aug


class UKTModelData(SkillModelData):
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
        user_question = sequences.question

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = UKTDataset(
            user_sequence,
            user_response,
            user_mask,
            questions=user_question,
            response_aug=build_ukt_response_aug(user_response, user_mask),
        )
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"UKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
