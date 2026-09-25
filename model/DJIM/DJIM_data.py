"""Framework-aligned skill sequences with observed answering behavior for DJIM."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from utils.model_data import SkillModelData
from utils.model_data.skill_model_data import WindowlateIterableDataset

# The article gives examples rather than a definitive per-dataset field list.
# These are genuine behavior columns retained by the framework's ASSIST data.
BEHAVIOR_COLUMNS = ("ms_first_response", "attempt_count", "hint_count")


def _require_behavior_columns(data: pl.DataFrame | pl.LazyFrame, source: str) -> None:
    columns = (
        data.collect_schema().names()
        if isinstance(data, pl.LazyFrame)
        else data.columns
    )
    missing = [column for column in BEHAVIOR_COLUMNS if column not in columns]
    if missing:
        raise ValueError(
            f"DJIM requires answering-behavior fields {BEHAVIOR_COLUMNS}; "
            f"{source} is missing {missing}. No placeholder values are used."
        )


def _behavior_array(data: pl.DataFrame, source: str) -> np.ndarray:
    _require_behavior_columns(data, source)
    if any(data[column].null_count() for column in BEHAVIOR_COLUMNS):
        raise ValueError(f"DJIM behavior fields contain null values in {source}")
    values = data.select(BEHAVIOR_COLUMNS).to_numpy().astype(np.float32)
    return np.log1p(values)


class DJIMDataset(Dataset):
    def __init__(self, skill, response, mask, question, behavior):
        self.skill = skill
        self.response = response
        self.mask = mask
        self.question = question
        self.behavior = behavior

    def __len__(self) -> int:
        return len(self.skill)

    def __getitem__(self, index: int):
        return (
            torch.as_tensor(self.skill[index], dtype=torch.long),
            torch.as_tensor(self.response[index], dtype=torch.long),
            torch.as_tensor(self.mask[index], dtype=torch.bool),
            torch.as_tensor(self.question[index], dtype=torch.long),
            torch.as_tensor(self.behavior[index], dtype=torch.float32),
        )


class DJIMWindowlateDataset(WindowlateIterableDataset):
    def __init__(self, parquet_path: str, max_seq_len: int, mean, std):
        super().__init__(parquet_path, max_seq_len)
        self.mean = mean
        self.std = std

    def _build_single_tensor(self, sample):
        base = super()._build_single_tensor(sample)
        values = np.stack(
            [sample[column].astype(np.float32) for column in BEHAVIOR_COLUMNS],
            axis=-1,
        )
        normalized = (np.log1p(values) - self.mean) / self.std
        # Target behavior is unavailable at prediction time. Its value is not
        # used by the shifted psychology state, and is cleared defensively.
        normalized[sample["mask"].astype(bool)] = 0
        behavior = np.zeros((self.max_seq_len, len(BEHAVIOR_COLUMNS)), dtype=np.float32)
        behavior[sample["position"]] = normalized
        return (*base, torch.from_numpy(behavior))


class DJIMModelData(SkillModelData):
    def prepare_data(self, rc: Any):
        if rc.data.fold < 0:
            raise ValueError("DJIM requires the framework's K-fold split")
        split = self.data_src.get_split_skill_sequence_data()
        if isinstance(split, pl.LazyFrame):
            split = split.collect()
        _require_behavior_columns(split, "split skill sequence")
        for column in ("sequence_id", "seq_pos", "skill", "question", "label", "fold"):
            if column not in split.columns:
                raise ValueError(f"DJIM requires preprocessed column {column!r}")
        window = self.data_src.get_windowlate_data()
        _require_behavior_columns(window, "windowlate data")

        values = _behavior_array(split, "split skill sequence")
        train_rows = (split["fold"].to_numpy() != rc.data.fold) & (
            split["fold"].to_numpy() != -1
        )
        mean = values[train_rows].mean(axis=0)
        std = values[train_rows].std(axis=0)
        std[std == 0] = 1
        values = (values - mean) / std

        skill, response, mask, _, question = self.build_sequence_data()
        behavior = np.zeros((*skill.shape, len(BEHAVIOR_COLUMNS)), dtype=np.float32)
        behavior[split["sequence_id"].to_numpy(), split["seq_pos"].to_numpy()] = values
        train, validation, _ = self.split_kfold_data(
            skill, response, mask, question, behavior, fold_idx=rc.data.fold
        )
        window_path = os.path.join(
            self.data_src.data_folder,
            f"{self.data_src.dataset}_windowlate.parquet",
        )
        window_dataset = DJIMWindowlateDataset(
            window_path, rc.data.max_seq_len, mean, std
        )
        test_loader = DataLoader(
            window_dataset,
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )
        return DJIMDataset(*train), DJIMDataset(*validation), test_loader
