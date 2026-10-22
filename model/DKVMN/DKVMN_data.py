from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class DKVMNDataset(Dataset):
    """DKVMN 数据集

    Args:
        sequences: 概念ID序列
        responses: 响应序列
        masks: 掩码序列
        late_group_ids: 题目级分组ID（可选，窗口测试时使用）
        true_labels: 真实标签序列（可选，窗口测试时使用）
    """

    def __init__(
        self, sequences, responses, masks, late_group_ids=None, true_labels=None
    ):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.late_group_ids = late_group_ids
        self.true_labels = true_labels
        self._is_window_mode = late_group_ids is not None and true_labels is not None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)

        if self._is_window_mode:
            late_group_id = torch.tensor(self.late_group_ids[idx], dtype=torch.long)
            true_labels = torch.tensor(self.true_labels[idx], dtype=torch.long)
            return sequence, response, mask, late_group_id, true_labels

        return sequence, response, mask


class DKVMNModelData(SkillModelData):
    """DKVMN 模型数据加载器

    Args:
        data_src: 数据源实例
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

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = DKVMNDataset(user_sequence, user_response, user_mask)
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"DKVMN data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
