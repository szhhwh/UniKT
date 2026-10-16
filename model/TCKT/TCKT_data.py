"""TCKT 模型数据处理模块。

- 题目序列 ``e``、答案序列 ``a``、掩码 ``mask``
- 响应时间桶序列 ``at``（由 ``ms_first_response`` 按秒离散化）
- 间隔时间桶序列 ``it``（由相邻 ``timestamp`` 之差按分钟离散化）
- 概念序列 ``c``（本次交互的主知识点）及完整技能列表
- 每题难度（贝叶斯估计，论文式 2，仅用训练折数据计算）

时间离散化遵循论文 4.4 节：“response time 按秒、interval time 按分钟”，并分别截断
到 ``max_rt_seconds`` / ``max_it_minutes`` 以滤除噪声长尾。
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import (
    PackedSequences,
    to_first_skill_ids,
    to_multi_skill_ids,
)

logger = get_logger(__name__)


class TCKTDataset(Dataset):
    """TCKT 数据集。

    每个样本返回 ``(e, at, a, it, c, mask, skills)``：
        e:    题目序列 [S]
        at:   响应时间桶序列 [S]
        a:    答案序列 [S]（同时作为输入特征与预测目标）
        it:   间隔时间桶序列 [S]
        c:    概念（主知识点）序列 [S]
        mask: 有效位置掩码 [S]
    """

    def __init__(self, e, at, a, it, c, mask, skills):
        self.e = e
        self.at = at
        self.a = a
        self.it = it
        self.c = c
        self.mask = mask
        self.skills = skills

    def __len__(self) -> int:
        return len(self.e)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.e[idx], dtype=torch.long),
            torch.tensor(self.at[idx], dtype=torch.long),
            torch.tensor(self.a[idx], dtype=torch.long),
            torch.tensor(self.it[idx], dtype=torch.long),
            torch.tensor(self.c[idx], dtype=torch.long),
            torch.tensor(self.mask[idx], dtype=torch.bool),
            torch.tensor(self.skills[idx], dtype=torch.long),
        )


class TCKTModelData(QuestionModelData):
    """TCKT 模型数据加载器。"""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的元信息。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            为含 ``difficulty``、``n_at``、
            ``n_it``、``num_questions``、``num_skills``、``max_seq_len`` 的字典。
        """
        max_seq_len = self.data_src.get_metadata("max_question_seq_len")
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        sequences = self.load_sequences(required=("ms_first_response", "timestamp"))
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask

        at_seq, it_seq, n_at, n_it = self._build_time_sequences(
            sequences, rc.model.max_rt_seconds, rc.model.max_it_minutes
        )

        c_seq = to_first_skill_ids(sequences.skills, offset=1)

        logger.info(
            f"TCKT data: num_questions={num_questions}, num_skills={num_skills}, "
            f"n_at={n_at}, n_it={n_it}, max_seq_len={max_seq_len}"
        )

        difficulty = self._compute_bayes_difficulty(
            user_sequence[partition.train],
            user_response[partition.train],
            user_mask[partition.train],
            num_questions,
        )

        dataset = TCKTDataset(
            user_sequence,
            at_seq,
            user_response,
            it_seq,
            c_seq,
            user_mask,
            to_multi_skill_ids(sequences.skills, padding_value=num_skills),
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"TCKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        info = {
            "difficulty": difficulty,
            "n_at": n_at,
            "n_it": n_it,
            "num_questions": num_questions,
            "num_skills": num_skills,
            "max_question_seq_len": max_seq_len,
        }
        return train_dataset, val_dataset, test_dataset, info

    def _build_time_sequences(
        self,
        sequences: PackedSequences,
        max_rt_seconds: int,
        max_it_minutes: int,
    ):
        """由 ``ms_first_response`` 与 ``timestamp`` 构建响应/间隔时间桶序列。

        Returns:
            at_seq: 响应时间桶 [N, S] ∈ [0, max_rt_seconds]
            it_seq: 间隔时间桶 [N, S] ∈ [0, max_it_minutes]
            n_at:   max_rt_seconds + 1
            n_it:   max_it_minutes + 1
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

        # Response time in seconds, truncated to [0, max_rt_seconds].
        rt_sec = np.floor(ms_grid / 1000.0)
        rt_sec = np.clip(rt_sec, 0, max_rt_seconds).astype(np.int32)

        # Interval time in minutes: adjacent timestamp diff per row, first position 0, truncated to [0, max_it_minutes].
        diff_ms = np.zeros_like(ts_grid)
        diff_ms[:, 1:] = ts_grid[:, 1:] - ts_grid[:, :-1]
        it_min = np.floor(diff_ms / 60000.0)
        it_min = np.clip(it_min, 0, max_it_minutes).astype(np.int32)

        n_at = int(max_rt_seconds) + 1
        n_it = int(max_it_minutes) + 1
        return rt_sec, it_min, n_at, n_it

    @staticmethod
    def _compute_bayes_difficulty(
        e: np.ndarray,
        a: np.ndarray,
        mask: np.ndarray,
        num_questions: int,
    ) -> np.ndarray:
        """基于贝叶斯估计的每题难度（式 2）。

            dif_q = NE_q/(NE_q+m) · D_q + m/(NE_q+m) · TD

        其中 D_q 为该题正确率，NE_q 为该题作答次数，m 为所有题的平均作答次数，
        TD 为整体正确率。仅用训练折交互计算；未见过的题难度取 TD。
        """
        valid = mask.astype(bool)
        flat_e = e[valid]
        flat_a = a[valid].astype(np.float64)

        count_inter = np.bincount(flat_e, minlength=num_questions).astype(np.float64)
        count_correct = np.bincount(
            flat_e, weights=flat_a, minlength=num_questions
        ).astype(np.float64)

        total_inter = float(count_inter.sum())
        total_correct = float(count_correct.sum())
        TD = total_correct / total_inter if total_inter > 0 else 0.5
        m = total_inter / num_questions if num_questions > 0 else 0.0

        with np.errstate(invalid="ignore", divide="ignore"):
            normal_diff = np.where(
                count_inter > 0, count_correct / np.maximum(count_inter, 1.0), 0.0
            )
            denom = count_inter + m
            bayes = np.where(
                denom > 0,
                count_inter / np.maximum(denom, 1e-12) * normal_diff
                + m / np.maximum(denom, 1e-12) * TD,
                TD,
            )
        return bayes.astype(np.float32)
