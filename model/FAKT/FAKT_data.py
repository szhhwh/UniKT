import os
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


def _compute_gaps_1d(skills: np.ndarray, timestamps: np.ndarray):
    """单条序列的逐位置 rgaps/sgaps/pcounts（未 clamp 的分桶索引）。

    公式：bucket = int(log2(Δ分钟) + 1)，首次出现/首位置 → 0。
    skills、timestamps 为按时间顺序的 1D 数组。
    """
    length = skills.shape[0]
    rgaps = np.zeros(length, dtype=np.int64)
    sgaps = np.zeros(length, dtype=np.int64)
    pcounts = np.zeros(length, dtype=np.int64)

    last_ts: dict[int, float] = {}
    skill_count: dict[int, int] = {}
    prev_ts = None

    for t in range(length):
        s = int(skills[t])
        ts = float(timestamps[t])

        cnt = skill_count.get(s, 0)
        pcounts[t] = int(np.log2(cnt) + 1) if cnt > 0 else 0

        if prev_ts is not None:
            dmin = (ts - prev_ts) / 60000.0  # ms -> minutes
            if dmin >= 1.0:
                sgaps[t] = int(np.log2(dmin) + 1)

        if s in last_ts:
            dmin = (ts - last_ts[s]) / 60000.0
            if dmin >= 1.0:
                rgaps[t] = int(np.log2(dmin) + 1)

        last_ts[s] = ts
        skill_count[s] = cnt + 1
        prev_ts = ts

    return rgaps, sgaps, pcounts


class FAKTDataset(Dataset):
    """FAKT 数据集（训练/验证）

    返回 7 元组：(sequence, response, mask, rgaps, sgaps, pcounts, question)
    """

    def __init__(self, sequences, responses, masks, rgaps, sgaps, pcounts, questions):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.rgaps = rgaps
        self.sgaps = sgaps
        self.pcounts = pcounts
        self.questions = questions

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        rgaps = torch.tensor(self.rgaps[idx], dtype=torch.long)
        sgaps = torch.tensor(self.sgaps[idx], dtype=torch.long)
        pcounts = torch.tensor(self.pcounts[idx], dtype=torch.long)
        question = torch.tensor(self.questions[idx], dtype=torch.long)
        return sequence, response, mask, rgaps, sgaps, pcounts, question


class FAKTWindowlateIterableDataset(WindowlateIterableDataset):
    """windowlate 流式数据集，按窗口计算时间间隔特征。

    返回 9 元组：(sequence, response, mask, late_group_id, label, question,
                  rgaps, sgaps, pcounts)
    时间间隔按窗口内真实时间戳计算（与训练同公式），并 clamp 到训练分桶数。
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

    def _build_single_tensor(self, sample):
        positions = sample["position"]
        max_seq_len = self.max_seq_len

        order = np.argsort(positions, kind="stable")
        pos_sorted = positions[order]
        skill_sorted = sample["skill"][order]
        ts_sorted = sample["timestamp"][order].astype(np.float64)

        rgaps_w, sgaps_w, pcounts_w = _compute_gaps_1d(skill_sorted, ts_sorted)
        rgaps_w = np.clip(rgaps_w, 0, max(self.num_rgap - 1, 0))
        sgaps_w = np.clip(sgaps_w, 0, max(self.num_sgap - 1, 0))
        pcounts_w = np.clip(pcounts_w, 0, max(self.num_pcount - 1, 0))

        sequence = np.zeros(max_seq_len, dtype=np.int64)
        response = np.zeros(max_seq_len, dtype=np.int64)
        mask = np.zeros(max_seq_len, dtype=np.bool_)
        late_group_id = np.full(max_seq_len, -1, dtype=np.int64)
        label = np.zeros(max_seq_len, dtype=np.int64)
        question = np.zeros(max_seq_len, dtype=np.int64)
        rgaps = np.zeros(max_seq_len, dtype=np.int64)
        sgaps = np.zeros(max_seq_len, dtype=np.int64)
        pcounts = np.zeros(max_seq_len, dtype=np.int64)

        sequence[pos_sorted] = skill_sorted
        response[pos_sorted] = sample["response"][order]
        mask[pos_sorted] = sample["mask"][order].astype(np.bool_)
        late_group_id[pos_sorted] = sample["group_id"][order]
        label[pos_sorted] = sample["true_label"][order]
        question[pos_sorted] = sample["question"][order]
        rgaps[pos_sorted] = rgaps_w
        sgaps[pos_sorted] = sgaps_w
        pcounts[pos_sorted] = pcounts_w

        return (
            torch.from_numpy(sequence),
            torch.from_numpy(response),
            torch.from_numpy(mask),
            torch.from_numpy(late_group_id),
            torch.from_numpy(label),
            torch.from_numpy(question),
            torch.from_numpy(rgaps),
            torch.from_numpy(sgaps),
            torch.from_numpy(pcounts),
        )


class FAKTModelData(SkillModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.num_rgap = 0
        self.num_sgap = 0
        self.num_pcount = 0

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练、验证和测试数据

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        data = self.load_split_data(required=("timestamp",))
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = to_first_skill_ids(sequences.skills)
        user_response = sequences.response
        user_mask = sequences.mask
        user_question = sequences.question

        rgaps, sgaps, pcounts = self._compute_time_gaps(sequences)

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = FAKTDataset(
            user_sequence,
            user_response,
            user_mask,
            rgaps,
            sgaps,
            pcounts,
            user_question,
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
            f"FAKT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}, "
            f"num_rgap={self.num_rgap}, num_sgap={self.num_sgap}, "
            f"num_pcount={self.num_pcount}"
        )

        return train_dataset, val_dataset, test_dataset

    @override
    def _create_windowlate_dataset(self) -> FAKTWindowlateIterableDataset:
        parquet_path = os.path.join(
            self.data_src.data_folder, f"{self.data_src.dataset}_windowlate.parquet"
        )
        return FAKTWindowlateIterableDataset(
            parquet_path=parquet_path,
            max_seq_len=self.data_src.get_metadata("max_windowlate_seq_len"),
            num_rgap=self.num_rgap,
            num_sgap=self.num_sgap,
            num_pcount=self.num_pcount,
        )

    def _compute_time_gaps(self, sequences):
        """Derive forgetting features from the already packed event history."""
        skills_mat = to_first_skill_ids(sequences.skills)
        ts_mat = sequences.features["timestamp"].astype(np.float64)
        num_users, max_seq_len = skills_mat.shape

        rgaps = np.zeros((num_users, max_seq_len), dtype=np.int64)
        sgaps = np.zeros((num_users, max_seq_len), dtype=np.int64)
        pcounts = np.zeros((num_users, max_seq_len), dtype=np.int64)
        for u in range(num_users):
            r, s, p = _compute_gaps_1d(skills_mat[u], ts_mat[u])
            rgaps[u] = r
            sgaps[u] = s
            pcounts[u] = p

        self.num_rgap = int(rgaps.max()) + 1
        self.num_sgap = int(sgaps.max()) + 1
        self.num_pcount = int(pcounts.max()) + 1

        logger.info(
            f"Computed time gaps: num_rgap={self.num_rgap}, "
            f"num_sgap={self.num_sgap}, num_pcount={self.num_pcount}"
        )

        return rgaps, sgaps, pcounts
