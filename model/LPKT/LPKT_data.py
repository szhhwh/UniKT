"""LPKT / LPKT-S 模型数据处理模块（两模型共享）。

- 题目序列 ``e``、答案序列 ``a``、掩码 ``mask``、用户 id ``uid``（LPKT-S 的
  student embedding 索引，LPKT 忽略）
- 答题用时 ``at``：``ms_first_response`` 毫秒→秒；间隔时间 ``it``：相邻
  ``timestamp`` 差→分钟、截断 ``max_it_minutes``。两者均为词表式离散化：
  全数据唯一值各一桶、id 从 1 起，0 为零向量 padding
- 每条交互的技能列表用于知识状态投影
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import PackedSequences, to_multi_skill_ids

logger = get_logger(__name__)


class LPKTDataset(Dataset):
    """LPKT / LPKT-S 数据集。

    每个样本返回 ``(e, at, a, it, mask, uid, skills)``：
        e:    题目序列 [S]
        at:   答题用时词表 id 序列 [S]（0 = padding 零向量）
        a:    答案序列 [S]（同时作为输入特征与预测目标）
        it:   间隔时间词表 id 序列 [S]（0 = padding 零向量）
        mask: 有效位置掩码 [S]
        uid:  用户（序列行）id 标量，LPKT-S 的 student embedding 索引；LPKT 忽略
    """

    def __init__(self, e, at, a, it, mask, uid, skills):
        self.e = e
        self.at = at
        self.a = a
        self.it = it
        self.mask = mask
        self.uid = uid
        self.skills = skills

    def __len__(self) -> int:
        return len(self.e)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.e[idx], dtype=torch.long),
            torch.tensor(self.at[idx], dtype=torch.long),
            torch.tensor(self.a[idx], dtype=torch.long),
            torch.tensor(self.it[idx], dtype=torch.long),
            torch.tensor(self.mask[idx], dtype=torch.bool),
            torch.tensor(self.uid[idx], dtype=torch.long),
            torch.tensor(self.skills[idx], dtype=torch.long),
        )


class LPKTModelData(QuestionModelData):
    """LPKT / LPKT-S 模型数据加载器（两模型共享）。"""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的元信息。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            为含 ``n_at``、``n_it``、``num_questions``、
            ``num_skills``、``num_users``、``max_seq_len`` 的字典。
        """
        max_seq_len = self.data_src.get_metadata("max_question_seq_len")
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")
        num_users = self.data_src.get_metadata("num_users")

        sequences = self.load_sequences(required=("ms_first_response", "timestamp"))
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        np.where(sequences.mask, sequences.user[:, None], 0)

        at_seq, it_seq, n_at, n_it = self._build_time_sequences(
            sequences, rc.model.max_it_minutes
        )

        uid = sequences.user

        logger.info(
            f"LPKT data: num_questions={num_questions}, num_skills={num_skills}, "
            f"num_users={num_users}, n_at={n_at}, n_it={n_it}, "
            f"max_seq_len={max_seq_len}"
        )

        dataset = LPKTDataset(
            user_sequence,
            at_seq,
            user_response,
            it_seq,
            user_mask,
            uid,
            to_multi_skill_ids(sequences.skills, padding_value=num_skills),
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"LPKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        info = {
            "n_at": n_at,
            "n_it": n_it,
            "num_questions": num_questions,
            "num_skills": num_skills,
            "num_users": num_users,
            "max_question_seq_len": max_seq_len,
        }
        return train_dataset, val_dataset, test_dataset, info

    def _build_time_sequences(
        self, sequences: PackedSequences, max_it_minutes: int
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """构建答题用时 / 间隔时间的词表 id 序列。

        词表对全数据统计唯一值（不分 fold——时间值不含标签信息）；有效位置
        （含值 0）映射到 ≥1，padding 位置恒为 0。

        Returns:
            at_seq / it_seq: 词表 id [N, S]
            n_at / n_it: 词表大小 = 唯一值数 + 1
        """
        ms_grid = np.nan_to_num(
            sequences.features["ms_first_response"].astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        ts_grid = np.nan_to_num(
            sequences.features["timestamp"].astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        valid = sequences.mask

        # Answer time in seconds; negative dirty values clipped to 0. No upper
        # cap — every distinct second keeps its own bucket (vocabulary style).
        at_sec = np.clip(np.floor(ms_grid / 1000.0), 0, None).astype(np.int64)

        # Interval time in minutes: adjacent diff per row, first position 0.
        diff_ms = np.zeros_like(ts_grid)
        diff_ms[:, 1:] = ts_grid[:, 1:] - ts_grid[:, :-1]
        it_min = np.clip(
            np.floor(np.clip(diff_ms, 0, None) / 60000.0), 0, max_it_minutes
        )
        it_min = it_min.astype(np.int64)

        at_vocab = np.unique(at_sec[valid])
        it_vocab = np.unique(it_min[valid])

        at_seq = np.zeros_like(at_sec, dtype=np.int32)
        it_seq = np.zeros_like(it_min, dtype=np.int32)
        at_seq[valid] = np.searchsorted(at_vocab, at_sec[valid]) + 1
        it_seq[valid] = np.searchsorted(it_vocab, it_min[valid]) + 1

        return at_seq, it_seq, int(at_vocab.size) + 1, int(it_vocab.size) + 1
