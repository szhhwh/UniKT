"""Skill-level model data module.

Provides the SkillModelData class for skill-based knowledge tracing model data
preparation, including sequence building, windowlate evaluation data loading,
and iterable dataset streaming from parquet files.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

if TYPE_CHECKING:
    import polars as pl

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import BaseModelData

logger = get_logger(__name__)


class WindowlateIterableDataset(IterableDataset):
    """Stream windowlate samples from a parquet file."""

    def __init__(
        self,
        parquet_path: str,
        max_seq_len: int,
        batch_read_rows: int = 200_000,
    ):
        """Initialise the iterable dataset.

        Args:
            parquet_path: Path to the parquet file containing windowlate data.
            max_seq_len: Maximum sequence length (window size).
            batch_read_rows: Number of rows to read per batch (default: 200000).
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.max_seq_len = max_seq_len
        self.batch_read_rows = batch_read_rows
        self._num_samples: int | None = None
        self._num_row_groups = None

    def _init_metadata(self) -> None:
        """Lazily initialise metadata from the parquet file."""
        if self._num_samples is None:
            import polars as pl

            stats = (
                pl.scan_parquet(self.parquet_path)
                .select(pl.col("sample_id").n_unique().alias("num_samples"))
                .collect(engine="streaming")
            )
            self._num_samples = int(stats["num_samples"][0])
            parquet_file = pq.ParquetFile(self.parquet_path)
            self._num_row_groups = parquet_file.num_row_groups

    def __len__(self) -> int:
        """Return the total number of samples."""
        self._init_metadata()
        assert self._num_samples is not None, "metadata init must set _num_samples"
        return self._num_samples

    def _build_single_tensor(
        self, sample: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, ...]:
        """Build a single sample tensor.

        Returns:
            Tuple of (sequence, response, mask, late_group_id, label, question,
            user_id) tensors, each of shape (max_seq_len,). user_id carries
            the ORIGINAL student id (the same id space as the split data's
            ``user`` column); trailing pad positions keep the buffer
            value 0 -- consumers must select valid positions via the mask or
            the window bounds, never via the raw user_id value. Subclasses
            may append extra feature tensors, so the arity is not fixed.
        """
        positions = sample["position"]

        sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        response = np.zeros(self.max_seq_len, dtype=np.int64)
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        late_group_id = np.full(self.max_seq_len, -1, dtype=np.int64)
        label = np.zeros(self.max_seq_len, dtype=np.int64)
        question = np.zeros(self.max_seq_len, dtype=np.int64)
        user_id = np.zeros(self.max_seq_len, dtype=np.int64)

        sequence[positions] = sample["skill"]
        response[positions] = sample["response"]
        mask[positions] = sample["mask"].astype(np.bool_)
        late_group_id[positions] = sample["group_id"]
        label[positions] = sample["true_label"]
        question[positions] = sample["question"]
        user_id[positions] = sample["user_id"]

        return (
            torch.from_numpy(sequence),
            torch.from_numpy(response),
            torch.from_numpy(mask),
            torch.from_numpy(late_group_id),
            torch.from_numpy(label),
            torch.from_numpy(question),
            torch.from_numpy(user_id),
        )

    def _read_batch_arrays(self, table: pa.Table) -> dict[str, np.ndarray]:
        """Read all columns of a Table as numpy arrays.

        Args:
            table: PyArrow Table.

        Returns:
            Dictionary mapping column names to numpy arrays.
        """
        return {col: table.column(col).to_numpy() for col in table.column_names}

    def _iter_row_groups(
        self, row_group_indices: list[int]
    ) -> Iterator[dict[str, np.ndarray]]:
        """Iterate over specified row groups, yielding batched data.

        Args:
            row_group_indices: List of row group indices to read.

        Yields:
            Dictionary of column name to numpy array for each row group.
        """
        parquet_file = pq.ParquetFile(self.parquet_path)

        for rg_idx in row_group_indices:
            table = parquet_file.read_row_group(rg_idx)
            yield self._read_batch_arrays(table)

    def _process_batch(
        self, batch: dict[str, np.ndarray]
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Process a batch of data, yielding individual samples.

        Args:
            batch: Dictionary of column arrays for one row group.

        Yields:
            Tensors for each sample in the batch.
        """
        sample_ids = batch["sample_id"]
        if sample_ids.size == 0:
            return

        boundaries = np.flatnonzero(sample_ids[1:] != sample_ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sample_ids.size]))

        keys = [k for k in batch if k != "sample_id"]

        for start, end in zip(starts, ends, strict=False):
            sample = {k: batch[k][start:end] for k in keys}
            yield self._build_single_tensor(sample)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Iterate over samples, with multi-worker data loading support."""
        self._init_metadata()
        assert self._num_row_groups is not None, (
            "metadata init must set _num_row_groups"
        )
        worker_info = get_worker_info()

        if worker_info is not None and worker_info.num_workers > 0:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            all_row_groups = list(range(self._num_row_groups))
            row_group_indices = all_row_groups[worker_id::num_workers]
        else:
            row_group_indices = list(range(self._num_row_groups))

        for batch in self._iter_row_groups(row_group_indices):
            yield from self._process_batch(batch)


class SkillModelData(BaseModelData):
    """Skill sequence data base class.

    Used to build skill-based (skill/concept) knowledge tracing model data.
    """

    def __init__(self, data_src: DataSource, cache: bool = False):
        """Initialise the skill-level model data object.

        Args:
            data_src: Data source object.
            cache: Whether to enable disk caching.
        """
        super().__init__(data_src, cache=cache)

    def _get_kfold_data(self) -> pl.DataFrame:
        """Override: retrieve K-fold labels from skill sequence data."""
        return self.data_src.get_split_skill_sequence_data()

    def build_sequence_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load user skill sequences from split skill sequence data.

        Returns:
            Tuple of (user_sequence, user_response, user_mask,
            user_id_sequence, user_question) as numpy arrays,
            each of shape (num_split_users, max_seq_len). user_id_sequence
            carries ORIGINAL student ids (from the ``user`` column),
            so one student's multiple splits share the same id -- matching
            the windowlate parquet's id space.
        """
        import numpy as np
        import polars as pl

        logger.info("Building skill sequences from split data...")

        data = self.data_src.get_split_skill_sequence_data()
        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_users = data["sequence_id"].n_unique()

        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)
        user_question = np.zeros((num_users, max_seq_len), dtype=int)

        user_indices = data["sequence_id"].to_numpy()
        seq_positions = data["seq_pos"].to_numpy()

        user_sequence[user_indices, seq_positions] = data["skill"].to_numpy()
        user_id_sequence[user_indices, seq_positions] = data["user"].to_numpy()
        user_response[user_indices, seq_positions] = data["label"].to_numpy()
        user_mask[user_indices, seq_positions] = 1
        user_question[user_indices, seq_positions] = data["question"].to_numpy()

        logger.debug(
            f"Built split skill sequences for {num_users} split users, max_len={max_seq_len}"
        )

        return user_sequence, user_response, user_mask, user_id_sequence, user_question

    def load_windowlate_data(
        self, max_seq_len: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Load windowlate evaluation samples.

        Loads sliding window data from a preprocessed parquet file and
        converts it to numpy arrays.

        Args:
            max_seq_len: Maximum sequence length (window size).

        Returns:
            Tuple of (user_sequence, user_response, user_mask,
            late_group_id, user_true_labels, user_question,
            user_id_sequence) as numpy arrays,
            each of shape (num_samples, max_seq_len). user_id_sequence
            (original student ids) is last, matching WindowlateIterableDataset.
        """
        import numpy as np

        data = self.data_src.get_windowlate_data()

        if data is None:
            raise ValueError(
                "No windowlate data available. Please re-run preprocessing with K-fold labels."
            )

        required_cols = [
            "sample_id",
            "position",
            "skill",
            "question",
            "response",
            "mask",
            "user_id",
            "group_id",
            "true_label",
        ]
        lazy_data = data.select(required_cols)
        all_data = lazy_data.collect(engine="streaming")
        sample_ids = all_data["sample_id"].to_numpy()
        positions = all_data["position"].to_numpy()
        num_samples = int(all_data["sample_id"].n_unique())

        if num_samples == 0:
            raise ValueError(
                "No windowlate data available. Please re-run preprocessing with K-fold labels."
            )

        user_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        user_response = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_mask = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_id_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        late_group_id = np.full((num_samples, max_seq_len), -1, dtype=np.int64)
        user_true_labels = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_question = np.zeros((num_samples, max_seq_len), dtype=np.int32)

        user_sequence[sample_ids, positions] = all_data["skill"].to_numpy()
        user_response[sample_ids, positions] = all_data["response"].to_numpy()
        user_mask[sample_ids, positions] = all_data["mask"].to_numpy()
        user_id_sequence[sample_ids, positions] = all_data["user_id"].to_numpy()
        late_group_id[sample_ids, positions] = all_data["group_id"].to_numpy()
        user_true_labels[sample_ids, positions] = all_data["true_label"].to_numpy()
        user_question[sample_ids, positions] = all_data["question"].to_numpy()

        logger.debug(
            f"Loaded windowlate data: samples={num_samples}, max_seq_len={max_seq_len}"
        )

        return (
            user_sequence,
            user_response,
            user_mask,
            late_group_id,
            user_true_labels,
            user_question,
            user_id_sequence,
        )

    def create_windowlate_iterable_dataset(
        self,
        max_seq_len: int,
        batch_read_rows: int = 200_000,
    ) -> WindowlateIterableDataset:
        """Create a WindowlateIterableDataset from the windowlate parquet file.

        Args:
            max_seq_len: Maximum sequence length (window size).
            batch_read_rows: Number of rows to read per batch (default: 200000).

        Returns:
            A configured WindowlateIterableDataset instance.
        """
        parquet_path = os.path.join(
            self.data_src.data_folder, f"{self.data_src.dataset}_windowlate.parquet"
        )

        return WindowlateIterableDataset(
            parquet_path=parquet_path,
            max_seq_len=max_seq_len,
            batch_read_rows=batch_read_rows,
        )
