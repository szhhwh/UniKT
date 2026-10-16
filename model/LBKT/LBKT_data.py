import numpy as np
import polars as pl
import torch
from scipy.stats import norm, poisson
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_multi_skill_ids

logger = get_logger(__name__)


class LBKTDataset(Dataset):
    def __init__(
        self,
        sequences,
        responses,
        masks,
        time_factors,
        attempt_factors,
        hint_factors,
        skills,
    ):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.time_factors = time_factors
        self.attempt_factors = attempt_factors
        self.hint_factors = hint_factors
        self.skills = skills

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.bool),
            torch.tensor(self.time_factors[index], dtype=torch.float32),
            torch.tensor(self.attempt_factors[index], dtype=torch.float32),
            torch.tensor(self.hint_factors[index], dtype=torch.float32),
            torch.tensor(self.skills[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class LBKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        data = self.load_split_data(
            required=("ms_first_response", "attempt_count", "hint_count")
        )
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        user_time_factor, user_attempt_factor, user_hint_factor = (
            self._build_behavior_factors(data, sequences, fold_idx=fold_idx)
        )

        dataset = LBKTDataset(
            user_sequence,
            user_response,
            user_mask,
            user_time_factor,
            user_attempt_factor,
            user_hint_factor,
            to_multi_skill_ids(
                sequences.skills, padding_value=self.data_src.get_metadata("num_skills")
            ),
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        return train_dataset, val_dataset, test_dataset

    def _build_behavior_factors(self, data: pl.DataFrame, sequences, fold_idx: int):
        """加载用户答题序列并计算行为因子。

        - time_factor: norm(mean_log, std_log).cdf(log(response_time))
          其中 mean_log/std_log 是每个 question 的 log(响应时间) 的统计量
        - attempt_factor: 1 - poisson.cdf(attempt - 1, mean_attempts)
        - hint_factor: 1 - poisson.cdf(hint - 1, mean_hints)

        所有 per-question 统计量仅基于训练折（fold != fold_idx 且 fold != -1）。
        """
        logger.info("Building response sequences with behavioral factors...")

        data = data.to_pandas()

        fold_labels = data["fold"].values
        train_mask = (fold_labels != fold_idx) & (fold_labels != -1)

        logger.info("Computing behavioral factors...")
        time_factors = self._compute_time_factor(data, train_mask)
        attempt_factors = self._compute_attempt_factor(data, train_mask)
        hint_factors = self._compute_hint_factor(data, train_mask)

        rows = np.searchsorted(sequences.sequence_id, data["sequence_id"].values)
        positions = data["seq_pos"].values
        factors = []
        for values in (time_factors, attempt_factors, hint_factors):
            packed = np.zeros(sequences.mask.shape, dtype=np.float32)
            packed[rows, positions] = values
            factors.append(packed)
        return tuple(factors)

    def _compute_time_factor(self, data, train_mask) -> np.ndarray:
        """计算时间因子。

        1. ms_first_response 毫秒转秒
        2. 仅对 response_time > 0 的行取 log
        3. 按 question 分组计算有效 log(time) 的 mean 和 std（仅训练折）
        4. time_factor = norm(mean, std).cdf(log(response_time))
        5. std == 0 时 time_factor = 1
        6. response_time == 0 的行（padding/缺失）time_factor = 1
        """
        response_times = data["ms_first_response"].values.astype(np.float64) / 1000.0

        time_factors = np.ones(len(data), dtype=np.float32)

        valid_mask = response_times > 0
        stats_mask = valid_mask & train_mask
        if not np.any(stats_mask):
            return time_factors

        valid_log_times = np.log(response_times[valid_mask])
        valid_questions = data.loc[valid_mask, "question"]

        stats_log_times = np.log(response_times[stats_mask])
        stats_questions = data.loc[stats_mask, "question"]

        stats_df = stats_questions.to_frame()
        stats_df["_log_time"] = stats_log_times
        question_stats = stats_df.groupby("question")["_log_time"].agg(["mean", "std"])
        question_stats["std"] = question_stats["std"].fillna(0)

        # Questions unseen in the training fold fall back to global training-fold statistics
        global_mean = float(stats_log_times.mean())
        global_std = float(stats_log_times.std())
        if np.isnan(global_std):
            global_std = 0.0

        # Vectorized: map-lookup per-row mean/std, then compute CDF in one pass
        means = valid_questions.map(question_stats["mean"]).values.astype(np.float64)
        stds = valid_questions.map(question_stats["std"]).values.astype(np.float64)

        missing = np.isnan(means)
        means[missing] = global_mean
        stds[missing] = global_std

        has_std = stds > 0
        if np.any(has_std):
            cdf_values = np.ones(len(valid_log_times), dtype=np.float32)
            cdf_values[has_std] = (
                norm(means[has_std], stds[has_std])
                .cdf(valid_log_times[has_std])
                .astype(np.float32)
            )
            time_factors[valid_mask] = cdf_values

        return time_factors

    def _compute_attempt_factor(self, data, train_mask) -> np.ndarray:
        """计算尝试因子。

        1 - poisson.cdf(attempt - 1, mean_attempts)
        其中 mean_attempts 为每个 question 的平均尝试次数（仅训练折）。
        """
        attempt_counts = data["attempt_count"].values.astype(np.float64)
        train_data = data.loc[train_mask]
        mean_attempts = train_data.groupby("question")["attempt_count"].mean()
        mean_per_row = data["question"].map(mean_attempts).values.astype(np.float64)

        # Questions unseen in the training fold fall back to the global mean attempt count
        global_mean = float(train_data["attempt_count"].mean())
        mean_per_row[np.isnan(mean_per_row)] = global_mean

        return (1 - poisson.cdf(attempt_counts - 1, mean_per_row)).astype(np.float32)

    def _compute_hint_factor(self, data, train_mask) -> np.ndarray:
        """计算提示因子。

        - mean_hints > 0 时：1 - poisson.cdf(hint - 1, mean_hints)
        - mean_hints == 0 时：hint_factor = 0
        mean_hints 为每个 question 的平均提示次数（仅训练折）。
        """
        hint_counts = data["hint_count"].values.astype(np.float64)
        train_data = data.loc[train_mask]
        mean_hints_series = train_data.groupby("question")["hint_count"].mean()
        mean_hints = data["question"].map(mean_hints_series).values.astype(np.float64)

        # Questions unseen in the training fold fall back to the global mean hint count
        global_mean = float(train_data["hint_count"].mean())
        mean_hints[np.isnan(mean_hints)] = global_mean

        hint_factors = np.zeros(len(data), dtype=np.float32)
        mask = mean_hints > 0
        if np.any(mask):
            hint_factors[mask] = (
                1 - poisson.cdf(hint_counts[mask] - 1, mean_hints[mask])
            ).astype(np.float32)

        return hint_factors
