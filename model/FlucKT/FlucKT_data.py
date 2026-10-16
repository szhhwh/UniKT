"""FlucKT 模型数据处理模块"""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class FlucKTDataset(Dataset):
    """FlucKT 数据集

    训练/验证模式:
        (sequence, response, mask, question)
    窗口测试模式:
        (sequence, response, mask, late_group_id, true_labels, question)
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
        self.sequences = torch.from_numpy(sequences).long()
        self.responses = torch.from_numpy(responses).long()
        self.masks = torch.from_numpy(masks).bool()
        self.questions = torch.from_numpy(questions).long()
        self.late_group_ids = late_group_ids
        self.true_labels = true_labels

        if late_group_ids is not None:
            self.late_group_ids = torch.from_numpy(late_group_ids).long()
        if true_labels is not None:
            self.true_labels = torch.from_numpy(true_labels).long()

        self._is_window_mode = late_group_ids is not None and true_labels is not None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        response = self.responses[idx]
        mask = self.masks[idx]
        question = self.questions[idx]

        if self._is_window_mode:
            late_group_id = self.late_group_ids[idx]
            true_labels = self.true_labels[idx]
            return sequence, response, mask, late_group_id, true_labels, question

        return sequence, response, mask, question


class FlucKTModelData(SkillModelData):
    """FlucKT 模型数据加载器

    继承自SkillModelData，使用技能序列构建方式，同时携带question序列
    供Rasch难度嵌入使用。

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

        dataset = FlucKTDataset(user_sequence, user_response, user_mask, user_question)
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"FlucKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
