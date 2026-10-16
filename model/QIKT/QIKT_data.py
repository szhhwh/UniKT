"""QIKT 模型数据处理模块"""

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_multi_skill_ids

logger = get_logger(__name__)


class QIKTDataset(Dataset):
    """QIKT 数据集

    提供问题序列、响应序列、掩码和多概念技能序列。

    Args:
        sequences: 问题ID序列 [N, S]
        responses: 响应序列 [N, S]
        masks: 掩码序列 [N, S]
        skills: 多概念技能ID序列 [N, S, max_concepts]，-1 表示填充
    """

    def __init__(self, sequences, responses, masks, skills):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.skills = skills

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.skills[idx], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class QIKTModelData(QuestionModelData):
    """QIKT 模型数据加载器

    从问题序列数据和问题-技能关系表中构建多概念技能序列。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.max_concepts = None

    @override
    def prepare_data(self, rc):
        """准备训练和验证数据

        Args:
            rc: RunConfig (OmegaConf DictConfig)

        Returns:
            (train_dataset, val_dataset, test_dataset) 元组
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask

        user_skills = to_multi_skill_ids(sequences.skills)
        self.max_concepts = user_skills.shape[-1]

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = QIKTDataset(user_sequence, user_response, user_mask, user_skills)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"QIKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
