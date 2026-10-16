"""CIKT 模型数据处理模块。

构建 CIKT 所需的全部输入：
    * 问题 / 作答 / 掩码序列（来自 ``统一序列打包``）；
    * 单概念序列 C（由本次交互的技能列表取首个技能得到）；
    * 题目难度查表 ``difficulty_table``（按训练交互正确率分箱，0=最难）；
    * 概念 → 候选题目表 ``concept_question_table``（用于 collate 阶段同概念随机替换 QR）。
"""

from functools import partial
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class CIKTDataset(Dataset):
    """CIKT 训练 / 验证 / 测试数据集。

    ``__getitem__`` 返回 ``(Q, Y, mask, C)``；同概念替换题 ``QR`` 由 collate_fn 注入。
    """

    def __init__(self, sequences, responses, masks, concepts):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.concepts = concepts

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.concepts[idx], dtype=torch.long),
        )


def cikt_collate_fn(
    batch, concept_question_table, concept_q_count, deterministic=False
):
    """批合并 + 同概念替换题 QR 采样。

    训练时对每个位置从其概念对应的候选题目中均匀采样一个；
    deterministic 模式（评估）固定取候选表第一个，保证推理可复现。
    """
    q, y, mask, c = default_collate(batch)
    count = concept_q_count[c]  # [B, L]
    if deterministic:
        idx = torch.zeros(c.shape, dtype=torch.long)
    else:
        rand = torch.rand(c.shape)
        idx = (rand * count.float()).long()
    idx = idx.clamp(min=0, max=concept_question_table.size(1) - 1)
    row = concept_question_table[c]  # [B, L, K]
    qr = row.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    valid = count > 0
    qr = torch.where(valid, qr, q)
    return q, y, mask, c, qr


class CIKTModelData(QuestionModelData):
    """CIKT 模型数据加载器。"""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    def _build_difficulty_table(
        self, data: pl.DataFrame, num_levels: int, fold_idx: int
    ) -> torch.Tensor:
        """题目难度分箱表。

        仅用训练交互（排除验证折与测试折）计算每题正确率，按正确率升序均分到
        ``num_levels`` 个等级，最低正确率（最难）=0。训练集中未出现的题目默认 0。
        """
        num_questions = self.data_src.get_metadata("num_questions")
        train = data.filter((pl.col("fold") != fold_idx) & (pl.col("fold") != -1))

        stats = (
            train.group_by("question")
            .agg(
                pl.col("label").sum().alias("correct"),
                pl.col("label").count().alias("n"),
            )
            .with_columns((pl.col("correct") / pl.col("n")).alias("acc"))
        )
        stats = stats.sort("acc")

        qids = stats["question"].to_numpy()
        n_q = len(qids)
        n_per = max(1, n_q // num_levels)
        table = np.zeros(num_questions, dtype=np.int64)
        for level in range(num_levels):
            start = level * n_per
            end = start + n_per if level < num_levels - 1 else n_q
            table[qids[start:end]] = level
        logger.info(
            f"CIKT difficulty table: {n_q} questions binned into {num_levels} levels "
        )
        return torch.as_tensor(table, dtype=torch.long)

    def _build_concept_question_table(self, q_matrix: np.ndarray):
        """概念 → 候选题目表（pad 到等长）与每概念真实候选数。"""
        num_skills = q_matrix.shape[1]
        concept_questions = []
        max_k = 1
        for s in range(num_skills):
            qs = np.sort(np.where(q_matrix[:, s] > 0)[0])
            concept_questions.append(qs)
            max_k = max(max_k, len(qs))
        table = np.zeros((num_skills, max_k), dtype=np.int64)
        counts = np.zeros(num_skills, dtype=np.int64)
        for s, qs in enumerate(concept_questions):
            if len(qs) == 0:
                continue
            padded = np.full(max_k, qs[0], dtype=np.int64)
            padded[: len(qs)] = qs
            table[s] = padded
            counts[s] = len(qs)
        return (
            torch.as_tensor(table, dtype=torch.long),
            torch.as_tensor(counts, dtype=torch.long),
        )

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据及 CIKT 专用查表。"""
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask

        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        user_concept = to_first_skill_ids(sequences.skills)

        difficulty_table = self._build_difficulty_table(
            data, rc.model.num_difficulty_levels, rc.data.fold
        )
        concept_question_table, concept_q_count = self._build_concept_question_table(
            q_matrix
        )

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = CIKTDataset(user_sequence, user_response, user_mask, user_concept)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        collate_fn = partial(
            cikt_collate_fn,
            concept_question_table=concept_question_table,
            concept_q_count=concept_q_count,
        )
        # Deterministic eval collate keeps val/test metrics reproducible.
        eval_collate_fn = partial(
            cikt_collate_fn,
            concept_question_table=concept_question_table,
            concept_q_count=concept_q_count,
            deterministic=True,
        )

        logger.info(
            f"CIKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            difficulty_table,
            collate_fn,
            eval_collate_fn,
        )
