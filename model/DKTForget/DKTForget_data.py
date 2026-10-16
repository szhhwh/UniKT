"""DKT-Forget 模型数据处理模块"""

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids
from utils.model_data.skill_model_data import WindowlateIterableDataset

logger = get_logger(__name__)


def compute_gap_features(
    skills: np.ndarray,
    timestamps: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算一条序列的 rgap/sgap/pcount 特征。

    Args:
        skills: 技能ID序列 [S] (int)
        timestamps: 时间戳序列 [S] (int, 毫秒)
        mask: 有效位置掩码 [S]；填充位（mask==0）跳过并置 0

    Returns:
        (rgap, sgap, pcount)，均为 int64 数组 [S]。

    说明:
        - rgap[t] = round(log2((ts[t]-ts[last_same_skill])/60000 + 1)) + 1，首次出现为 0
        - sgap[t] = round(log2((ts[t]-ts[prev])/60000 + 1)) + 1，序列首位为 0
        - pcount[t] = round(log2(prior_count_of_skill + 1))
    """
    seq_len = len(skills)
    rgap = np.zeros(seq_len, dtype=np.int64)
    sgap = np.zeros(seq_len, dtype=np.int64)
    pcount = np.zeros(seq_len, dtype=np.int64)
    dlast: dict[int, int] = {}  # skill 上次时间戳
    dcount: dict[int, int] = {}  # skill 此前出现次数
    prev_t: int | None = None
    for i in range(seq_len):
        if not mask[i]:
            continue
        s = int(skills[i])
        t = int(timestamps[i])
        if s in dlast:
            minutes = (t - dlast[s]) / 60000.0
            rgap[i] = round(math.log(minutes + 1.0, 2)) + 1
        dlast[s] = t
        if prev_t is not None:
            minutes = (t - prev_t) / 60000.0
            sgap[i] = round(math.log(minutes + 1.0, 2)) + 1
        prev_t = t
        cnt = dcount.get(s, 0)
        pcount[i] = round(math.log(cnt + 1.0, 2))
        dcount[s] = cnt + 1
    return rgap, sgap, pcount


class DKTForgetDataset(Dataset):
    """DKT-Forget 训练/验证数据集

    Args:
        sequences: 技能ID序列 [N, S]
        responses: 响应序列 [N, S]
        masks: 掩码序列 [N, S]
        rgaps/sgaps/pcounts: 遗忘特征序列 [N, S]
    """

    def __init__(self, sequences, responses, masks, rgaps, sgaps, pcounts):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.rgaps = rgaps
        self.sgaps = sgaps
        self.pcounts = pcounts

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.rgaps[idx], dtype=torch.long),
            torch.tensor(self.sgaps[idx], dtype=torch.long),
            torch.tensor(self.pcounts[idx], dtype=torch.long),
        )


class DKTForgetWindowlateIterableDataset(WindowlateIterableDataset):
    def __init__(
        self,
        parquet_path: str,
        max_seq_len: int,
        num_rgap: int,
        num_sgap: int,
        num_pcount: int,
    ):
        super().__init__(parquet_path, max_seq_len)
        self.num_rgap = num_rgap
        self.num_sgap = num_sgap
        self.num_pcount = num_pcount

    @override
    def _build_single_tensor(self, sample: dict[str, np.ndarray]):
        positions = sample["position"]
        max_seq_len = self.max_seq_len

        sequence = np.zeros(max_seq_len, dtype=np.int64)
        response = np.zeros(max_seq_len, dtype=np.int64)
        mask = np.zeros(max_seq_len, dtype=np.bool_)
        late_group_id = np.full(max_seq_len, -1, dtype=np.int64)
        label = np.zeros(max_seq_len, dtype=np.int64)
        question = np.zeros(max_seq_len, dtype=np.int64)
        timestamp = np.zeros(max_seq_len, dtype=np.int64)

        sequence[positions] = sample["skill"]
        response[positions] = sample["response"]
        mask[positions] = sample["mask"].astype(np.bool_)
        late_group_id[positions] = sample["group_id"]
        label[positions] = sample["true_label"]
        question[positions] = sample["question"]
        timestamp[positions] = sample["timestamp"]

        # 窗口内计算遗忘特征
        rgap, sgap, pcount = compute_gap_features(sequence, timestamp, mask)
        rgap = np.clip(rgap, 0, self.num_rgap - 1)
        sgap = np.clip(sgap, 0, self.num_sgap - 1)
        pcount = np.clip(pcount, 0, self.num_pcount - 1)

        return (
            torch.from_numpy(sequence),
            torch.from_numpy(response),
            torch.from_numpy(mask),
            torch.from_numpy(late_group_id),
            torch.from_numpy(label),
            torch.from_numpy(question),
            torch.from_numpy(rgap),
            torch.from_numpy(sgap),
            torch.from_numpy(pcount),
        )


class DKTForgetModelData(SkillModelData):
    """DKT-Forget 模型数据加载器"""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.num_rgap: int = 0
        self.num_sgap: int = 0
        self.num_pcount: int = 0

    def _build_gap_features(self, sequences):
        user_sequence = to_first_skill_ids(sequences.skills)
        user_mask = sequences.mask
        user_timestamp = sequences.features["timestamp"]
        num_sequences, max_seq_len = user_sequence.shape
        rgaps = np.zeros_like(user_sequence)
        sgaps = np.zeros_like(user_sequence)
        pcounts = np.zeros_like(user_sequence)

        max_rgap = max_sgap = max_pcount = 0
        for u in range(num_sequences):
            m = user_mask[u].astype(bool)
            if not m.any():
                continue
            r, s, p = compute_gap_features(user_sequence[u], user_timestamp[u], m)
            rgaps[u], sgaps[u], pcounts[u] = r, s, p
            max_rgap = max(max_rgap, int(r.max()))
            max_sgap = max(max_sgap, int(s.max()))
            max_pcount = max(max_pcount, int(p.max()))

        # 词表大小 = 全局最大值 + 1
        self.num_rgap = max_rgap + 1
        self.num_sgap = max_sgap + 1
        self.num_pcount = max_pcount + 1
        logger.info(
            f"DKTForget gap vocab: num_rgap={self.num_rgap} "
            f"num_sgap={self.num_sgap} num_pcount={self.num_pcount}"
        )

        return rgaps, sgaps, pcounts

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练、验证和测试数据"""
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        sequences = self.load_sequences(required=("timestamp",))
        partition = sequences.partition(rc.data.fold)
        user_sequence = to_first_skill_ids(sequences.skills)
        user_response = sequences.response
        user_mask = sequences.mask
        rgaps, sgaps, pcounts = self._build_gap_features(sequences)

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = DKTForgetDataset(
            user_sequence, user_response, user_mask, rgaps, sgaps, pcounts
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
            f"DKTForget data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    @override
    def _create_windowlate_dataset(self) -> DKTForgetWindowlateIterableDataset:
        import os

        return DKTForgetWindowlateIterableDataset(
            parquet_path=os.path.join(
                self.data_src.data_folder, f"{self.data_src.dataset}_windowlate.parquet"
            ),
            max_seq_len=self.data_src.get_metadata("max_windowlate_seq_len"),
            num_rgap=self.num_rgap,
            num_sgap=self.num_sgap,
            num_pcount=self.num_pcount,
        )
