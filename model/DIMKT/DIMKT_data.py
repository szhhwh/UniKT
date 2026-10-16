"""DIMKT 模型数据处理模块"""

from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class DIMKTDataset(Dataset):
    """DIMKT 训练 / 验证数据集。"""

    def __init__(self, sequences, responses, masks, questions):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        question = torch.tensor(self.questions[idx], dtype=torch.long)
        return sequence, response, mask, question


class DIMKTModelData(SkillModelData):
    """DIMKT 模型数据加载器。

    Args:
        data_src: 数据源实例，包含原始数据。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    def compute_difficulty_tables(
        self,
        data: pl.DataFrame,
        difficult_levels: int,
        fold_idx: int,
        min_attempts: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算技能 / 题目的离散难度等级查表。

        仅使用训练交互来计算正确率。对于训练集中未出现的 skill/question，
        查表返回默认难度 1。

        参数:
            difficult_levels: 难度离散化等级数 D。
            fold_idx: 当前 K-fold 的验证折索引；该折与测试折 (fold=-1) 都会被排除。
            min_attempts: 参与统计的最小样本数阈值（默认 30）。

        返回:
            (skill_diff_table, question_diff_table):
                - skill_diff_table: long tensor, shape=[num_skills]，``table[sid]`` 为难度等级。
                - question_diff_table: long tensor, shape=[num_questions]，``table[qid]`` 为难度等级。
                等级取值范围 [1, D+1]；未参与统计的 id 默认为 1。
        """
        num_skills = self.data_src.get_metadata("num_skills")
        num_questions = self.data_src.get_metadata("num_questions")

        train_data = data.filter((pl.col("fold") != fold_idx) & (pl.col("fold") != -1))

        skill_diff = self._discretize_difficulty(
            train_data, "skill", num_skills, difficult_levels, min_attempts
        )
        question_diff = self._discretize_difficulty(
            train_data, "question", num_questions, difficult_levels, min_attempts
        )

        logger.info(
            f"DIMKT difficulty tables: skills={num_skills}, questions={num_questions}, "
            f"difficult_levels={difficult_levels}, min_attempts={min_attempts}"
        )

        return (
            torch.as_tensor(skill_diff, dtype=torch.long),
            torch.as_tensor(question_diff, dtype=torch.long),
        )

    @staticmethod
    def _discretize_difficulty(
        data,
        key: str,
        num_items: int,
        difficult_levels: int,
        min_attempts: int,
    ) -> np.ndarray:
        stats = data.group_by(key).agg(
            pl.col("label").count().alias("n"),
            pl.col("label").sum().alias("correct"),
        )
        rate = pl.col("correct") / pl.col("n")
        level = (rate * difficult_levels).floor().cast(pl.Int64) + 1
        level = (
            pl.when((pl.col("n") < min_attempts) | (pl.col("correct") == 0))
            .then(pl.lit(1))
            .otherwise(level)
            .alias("level")
        )
        stats = stats.select([key, level])

        # Default difficulty level 1 for ids absent from the training split
        table = np.ones(num_items, dtype=np.int64)
        ids = stats[key].to_numpy()
        levels = stats["level"].to_numpy()
        valid = (ids >= 0) & (ids < num_items)
        table[ids[valid]] = levels[valid]
        return table

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练、验证与窗口评估数据。"""
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

        dataset = DIMKTDataset(user_sequence, user_response, user_mask, user_question)
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"DIMKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test(window)={len(test_dataset)}"
        )

        skill_diff_table, question_diff_table = self.compute_difficulty_tables(
            data, rc.model.difficult_levels, rc.data.fold
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            skill_diff_table,
            question_diff_table,
        )
