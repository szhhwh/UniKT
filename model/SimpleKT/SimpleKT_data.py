"""SimpleKT 模型数据处理模块"""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class SimpleKTDataset(Dataset):
    """SimpleKT 数据集

    处理输入数据，将其转换为模型所需的格式。

    Args:
        sequences: 技能ID序列
        responses: 响应序列
        masks: 掩码序列
        questions: 题目ID序列
        late_group_ids: 题目级分组ID（可选，窗口测试时使用）
        true_labels: 真实标签序列（可选，窗口测试时使用）

    Returns:
        训练/验证模式：
            (sequence, response, mask, question) 元组
        窗口测试模式：
            (sequence, response, mask, late_group_id, true_labels, question) 元组
    """

    def __init__(
        self,
        sequences,
        responses,
        masks,
        questions,
        late_group_ids=None,
        true_labels=None,
    ):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions
        self.late_group_ids = late_group_ids
        self.true_labels = true_labels

        self._is_window_mode = late_group_ids is not None and true_labels is not None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        question = torch.tensor(self.questions[idx], dtype=torch.long)

        if self._is_window_mode:
            late_group_id = torch.tensor(self.late_group_ids[idx], dtype=torch.long)
            true_labels = torch.tensor(self.true_labels[idx], dtype=torch.long)
            return sequence, response, mask, late_group_id, true_labels, question

        return sequence, response, mask, question


class SimpleKTModelData(SkillModelData):
    """SimpleKT 模型数据加载器

    负责处理数据的预处理和加载，为 SimpleKT 模型提供训练和验证数据。
    继承自 SkillModelData，使用技能序列构建方式。

    Args:
        data_src: 数据源实例，包含原始数据
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练和验证数据

        Args:
            rc: RunConfig (OmegaConf DictConfig)

        Returns:
            训练数据集、验证数据集和窗口验证数据集
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = to_first_skill_ids(sequences.skills)
        user_response = sequences.response
        user_mask = sequences.mask
        user_question = sequences.question

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = SimpleKTDataset(
            user_sequence, user_response, user_mask, user_question
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
            f"SimpleKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
