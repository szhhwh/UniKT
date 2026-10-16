"""MCSKT 模型数据处理模块

在标准技能序列（``SkillModelData``）基础上，额外计算三类遗忘特征
（repeated time gap / sequence time gap / past trial count，均 log2 离散化），
对应论文 Eq.2 的遗忘特征 f。

数据格式：
    训练/验证: (sequence, response, mask, question, rgap, sgap, pcount)  7-元组
    窗口测试:  (sequence, response, mask, late_group_id, true_labels, question,
               rgap, sgap, pcount)  9-元组（遗忘特征由窗口内时间戳实时计算）
"""

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


def compute_time_gaps(
    skills: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int,
    num_rgap: int,
    num_sgap: int,
    num_pcount: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算单个序列的三类遗忘特征（log2 离散化）。"""
    max_len = len(skills)
    rgap = np.zeros(max_len, dtype=np.int64)
    sgap = np.full(max_len, num_sgap - 1, dtype=np.int64)
    pcount = np.zeros(max_len, dtype=np.int64)

    last_ts: dict[int, float] = {}
    skill_count: dict[int, int] = {}

    def _log2_bucket(val: float) -> int:
        return round(math.log(val + 1, 2))

    for t in range(seq_len):
        s = int(skills[t])

        # rgap: minutes since the last occurrence of the same concept
        if s in last_ts and timestamps[t] > 0 and last_ts[s] > 0:
            diff_min = (timestamps[t] - last_ts[s]) / 60000.0
            if diff_min >= 0:
                rgap[t] = min(_log2_bucket(diff_min) + 1, num_rgap - 1)

        # sgap: minutes since the previous interaction
        if t > 0 and timestamps[t] > 0 and timestamps[t - 1] > 0:
            diff_min = (timestamps[t] - timestamps[t - 1]) / 60000.0
            if diff_min >= 0:
                sgap[t] = min(_log2_bucket(diff_min) + 1, num_sgap - 1)

        # pcount: prior practice count for this concept
        if s in skill_count:
            pcount[t] = min(_log2_bucket(skill_count[s]), num_pcount - 1)

        last_ts[s] = timestamps[t]
        skill_count[s] = skill_count.get(s, 0) + 1

    return rgap, sgap, pcount


class MCSKTDataset(Dataset):
    """MCSKT 训练/验证数据集（7-元组）。"""

    def __init__(self, sequences, responses, masks, questions, rgaps, sgaps, pcounts):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions
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
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.rgaps[idx], dtype=torch.long),
            torch.tensor(self.sgaps[idx], dtype=torch.long),
            torch.tensor(self.pcounts[idx], dtype=torch.long),
        )


class MCSKTWindowlateIterableDataset(WindowlateIterableDataset):
    """带遗忘特征的 windowlate 流式数据集。

    在标准 6-元组基础上按窗口实时计算 rgap/sgap/pcount，
    返回 9-元组 ``(sequence, response, mask, late_group_id, label, question,
    rgap, sgap, pcount)``。
    """

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

    def _build_single_tensor(self, sample: dict[str, np.ndarray]):
        """构建单样本 9-元组，含按窗口计算的遗忘特征。"""
        positions = sample["position"]

        sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        response = np.zeros(self.max_seq_len, dtype=np.int64)
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        late_group_id = np.full(self.max_seq_len, -1, dtype=np.int64)
        label = np.zeros(self.max_seq_len, dtype=np.int64)
        question = np.zeros(self.max_seq_len, dtype=np.int64)
        rgap = np.zeros(self.max_seq_len, dtype=np.int64)
        sgap = np.full(self.max_seq_len, self.num_sgap - 1, dtype=np.int64)
        pcount = np.zeros(self.max_seq_len, dtype=np.int64)

        sequence[positions] = sample["skill"]
        response[positions] = sample["response"]
        mask[positions] = sample["mask"].astype(np.bool_)
        late_group_id[positions] = sample["group_id"]
        label[positions] = sample["true_label"]
        question[positions] = sample["question"]

        seq_len = int(positions[-1]) + 1 if len(positions) > 0 else 0
        if seq_len > 0:
            r, s, p = compute_time_gaps(
                sequence[:seq_len],
                sample["timestamp"].astype(np.float64),
                seq_len,
                self.num_rgap,
                self.num_sgap,
                self.num_pcount,
            )
            rgap[:seq_len] = r
            sgap[:seq_len] = s
            pcount[:seq_len] = p

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


class MCSKTModelData(SkillModelData):
    """MCSKT 数据加载器：技能序列 + 遗忘特征。"""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    def _build_time_gaps(
        self,
        user_sequence: np.ndarray,
        user_mask: np.ndarray,
        user_timestamp: np.ndarray,
        num_rgap: int,
        num_sgap: int,
        num_pcount: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_users = user_sequence.shape[0]
        max_seq_len = user_sequence.shape[1]

        user_rgap = np.zeros((num_users, max_seq_len), dtype=np.int64)
        user_sgap = np.full((num_users, max_seq_len), num_sgap - 1, dtype=np.int64)
        user_pcount = np.zeros((num_users, max_seq_len), dtype=np.int64)

        for i in range(num_users):
            mask_row = user_mask[i]
            valid_positions = np.where(mask_row > 0)[0]
            if len(valid_positions) == 0:
                continue
            seq_len = int(valid_positions[-1]) + 1

            r, s, p = compute_time_gaps(
                user_sequence[i],
                user_timestamp[i],
                seq_len,
                num_rgap,
                num_sgap,
                num_pcount,
            )
            user_rgap[i] = r
            user_sgap[i] = s
            user_pcount[i] = p

        return user_rgap, user_sgap, user_pcount

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        self.num_rgap = rc.model.num_rgap
        self.num_sgap = rc.model.num_sgap
        self.num_pcount = rc.model.num_pcount

        data = self.load_split_data(required=("timestamp",))
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = to_first_skill_ids(sequences.skills)
        user_response = sequences.response
        user_mask = sequences.mask
        user_question = sequences.question

        user_timestamp = sequences.features["timestamp"].astype(np.float64)
        logger.info("Computing forgetting features (rgap, sgap, pcount)...")
        user_rgap, user_sgap, user_pcount = self._build_time_gaps(
            user_sequence,
            user_mask,
            user_timestamp,
            self.num_rgap,
            self.num_sgap,
            self.num_pcount,
        )

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = MCSKTDataset(
            user_sequence,
            user_response,
            user_mask,
            user_question,
            user_rgap,
            user_sgap,
            user_pcount,
        )
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self._create_windowlate_dataset()

        logger.info(
            f"MCSKT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    @override
    def _create_windowlate_dataset(self) -> MCSKTWindowlateIterableDataset:
        import os

        return MCSKTWindowlateIterableDataset(
            parquet_path=os.path.join(
                self.data_src.data_folder, f"{self.data_src.dataset}_windowlate.parquet"
            ),
            max_seq_len=self.data_src.get_metadata("max_windowlate_seq_len"),
            num_rgap=self.num_rgap,
            num_sgap=self.num_sgap,
            num_pcount=self.num_pcount,
        )
