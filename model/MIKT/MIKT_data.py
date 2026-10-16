"""MIKT 模型数据处理模块"""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_multi_skill_ids

logger = get_logger(__name__)


class MIKTDataset(Dataset):
    """MIKT 数据集

    返回 (sequence, response, mask) 元组。
    """

    def __init__(self, sequences, responses, masks, skills):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.skills = skills

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.bool),
            torch.tensor(self.skills[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class MIKTModelData(QuestionModelData):
    """MIKT 模型数据加载器

    使用问题级序列数据，构建问题-技能关联矩阵供模型使用。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练和验证数据

        Args:
            rc: RunConfig (OmegaConf DictConfig)

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask

        skills = to_multi_skill_ids(
            sequences.skills, padding_value=self.data_src.get_metadata("num_skills")
        )

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = MIKTDataset(user_sequence, user_response, user_mask, skills)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.debug(
            f"MIKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
