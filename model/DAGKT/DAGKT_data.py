from typing import Any

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_multi_skill_ids

logger = get_logger(__name__)


class DAGKTDataset(Dataset):
    """DAGKT samples: (sequence, response, mask, attempt_counts, skills)."""

    def __init__(self, sequences, responses, masks, attempt_counts, skills):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.responses = torch.as_tensor(responses, dtype=torch.long)
        self.masks = torch.as_tensor(masks, dtype=torch.long)
        self.skills = torch.as_tensor(skills, dtype=torch.long)
        self.attempt_counts = torch.as_tensor(attempt_counts, dtype=torch.float)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.responses[index],
            self.masks[index],
            self.attempt_counts[index],
            self.skills[index],
        )

    def __len__(self):
        return len(self.sequences)


class DAGKTModelData(QuestionModelData):
    """DAGKT 模型数据准备类。

    在 GIKT 数据基础上增加：
    - 题目正确率（difficulty_rates）: 从训练集统计每题的正确率
    - 学生尝试次数（attempt_counts）: 直接使用数据集中的 attempt_count 字段
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any):
        r"""准备 DAGKT 模型所需的数据。"""
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        sequences = self.load_sequences(optional=("attempt_count",))
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        np.where(sequences.mask, sequences.user[:, None], 0)
        attempt_counts = sequences.features.get(
            "attempt_count", np.zeros_like(user_response)
        ).astype(np.float32)

        graph = self.build_hetero_graph(
            [
                (
                    "question",
                    "has",
                    "skill",
                )
            ]
        )

        num_questions = self.data_src.get_metadata("num_questions")
        question_difficulty = self._compute_question_correct_rates(
            num_questions, fold_idx
        )

        attempt_counts = self._normalize_attempt_counts(attempt_counts)

        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = DAGKTDataset(
            user_sequence,
            user_response,
            user_mask,
            attempt_counts,
            to_multi_skill_ids(
                sequences.skills, padding_value=self.data_src.get_metadata("num_skills")
            ),
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            graph,
            question_difficulty,
        )

    def _compute_question_correct_rates(
        self, num_questions: int, fold_idx: int | None = None
    ) -> torch.Tensor:
        """仅从训练集统计每题的正确率。

        排除验证集（fold == fold_idx）和测试集（fold == -1），
        只使用训练集数据以避免数据泄露。

        Args:
            num_questions: 题目总数
            fold_idx: 当前验证集的 fold 索引（可选）

        Returns:
            题目正确率张量 [num_questions, 1]
        """
        import tqdm.auto as tqdm

        data = self.data_src.get_sequence_data().to_pandas()

        if "fold" in data.columns:
            mask = data["fold"] != -1
            if fold_idx is not None:
                mask &= data["fold"] != fold_idx
            data = data[mask]
            logger.info(
                f"Computing correct rates from training data only "
                f"(excluded fold {fold_idx} and test fold -1)."
            )

        question_stats = data.groupby("question")["label"].mean()

        # Default 0.5 for questions absent from the training split
        correct_rates = torch.ones(num_questions, 1) * 0.5
        for qid, rate in tqdm.tqdm(
            question_stats.items(),
            total=len(question_stats),
            desc="Computing question correct rates",
        ):
            if qid < num_questions:
                correct_rates[qid] = rate

        logger.info(
            f"Question correct rates: min={correct_rates.min().item():.4f}, "
            f"max={correct_rates.max().item():.4f}, "
            f"mean={correct_rates.mean().item():.4f}"
        )

        return correct_rates

    def _normalize_attempt_counts(self, attempt_counts: np.ndarray) -> np.ndarray:
        """归一化 attempt_count 到 [0, 1]。

        Args:
            attempt_counts: 原始尝试次数 [num_users, max_seq_len]

        Returns:
            归一化的尝试次数 [num_users, max_seq_len]
        """
        max_attempts = attempt_counts.max()
        if max_attempts > 0:
            attempt_counts = attempt_counts / max_attempts

        logger.info(
            f"Attempt counts: max_raw={max_attempts}, "
            f"normalized_max={attempt_counts.max():.4f}"
        )

        return attempt_counts
