"""Skill-level model data module.

Provides the SkillModelData class for skill-based knowledge tracing model data
preparation, including sequence building, windowlate evaluation data loading,
and iterable dataset streaming from parquet files.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import (
    IterableDataset,
    default_collate,
    get_worker_info,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

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
    ):
        """Initialise the iterable dataset.

        Args:
            parquet_path: Path to the parquet file containing windowlate data.
            max_seq_len: Maximum sequence length (window size).
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.max_seq_len = max_seq_len
        self._num_samples: int | None = None
        self._num_row_groups = None

    def _init_metadata(self) -> None:
        """Lazily initialise metadata from the parquet file."""
        if self._num_samples is None:
            parquet_file = pq.ParquetFile(self.parquet_path)
            self._num_row_groups = parquet_file.num_row_groups
            sample_ids = parquet_file.read(columns=["sample_id"]).column("sample_id")
            self._num_samples = int(
                np.unique(sample_ids.to_numpy(zero_copy_only=False)).size
            )

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
        self, row_group_indices: Iterable[int]
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

    @staticmethod
    def _split_samples(
        batch: dict[str, np.ndarray],
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield sample views without allocating padded tensors."""
        sample_ids = batch["sample_id"]
        if sample_ids.size == 0:
            return

        boundaries = np.flatnonzero(sample_ids[1:] != sample_ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sample_ids.size]))

        keys = [k for k in batch if k != "sample_id"]

        for start, end in zip(starts, ends, strict=False):
            yield {k: batch[k][start:end] for k in keys}

    def _iter_worker_arrays(self) -> Iterator[dict[str, np.ndarray]]:
        """Read this worker's row groups in their original order."""
        self._init_metadata()
        assert self._num_row_groups is not None, (
            "metadata init must set _num_row_groups"
        )
        worker_info = get_worker_info()

        row_group_indices = range(self._num_row_groups)
        if worker_info is not None:
            row_group_indices = row_group_indices[
                worker_info.id :: worker_info.num_workers
            ]

        yield from self._iter_row_groups(row_group_indices)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Iterate over padded samples, preserving the single-sample interface."""
        for batch in self._iter_worker_arrays():
            for sample in self._split_samples(batch):
                yield self._build_single_tensor(sample)

    def _collate_batch(
        self, samples: list[dict[str, np.ndarray]]
    ) -> list[torch.Tensor]:
        """Allocate and scatter a full batch, retaining custom sample builders."""
        if (
            type(self)._build_single_tensor
            is not WindowlateIterableDataset._build_single_tensor
        ):
            # Model-specific builders own their extra features and tuple order.
            return default_collate([self._build_single_tensor(s) for s in samples])

        positions = np.concatenate([s["position"] for s in samples])
        rows = np.repeat(np.arange(len(samples)), [len(s["position"]) for s in samples])
        fields = (
            "skill",
            "response",
            "mask",
            "group_id",
            "true_label",
            "question",
            "user_id",
        )
        tensors = []
        in_worker = get_worker_info() is not None
        for field in fields:
            dtype = torch.bool if field == "mask" else torch.int64
            tensor = torch.empty(
                (len(samples), self.max_seq_len), dtype=dtype, device="cpu"
            )
            if in_worker:
                tensor.share_memory_()
            array = tensor.numpy()
            array.fill(-1 if field == "group_id" else 0)
            array[rows, positions] = np.concatenate([s[field] for s in samples])
            tensors.append(tensor)
        return tensors


class _WindowlateBatchDataset(WindowlateIterableDataset):
    """Expose raw sample views to the batch collator without changing the source."""

    def __init__(self, source: WindowlateIterableDataset):
        super().__init__(source.parquet_path, source.max_seq_len)
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self) -> Iterator[Any]:
        for arrays in self.source._iter_worker_arrays():
            yield from self.source._split_samples(arrays)


class SkillModelData(BaseModelData):
    """Skill sequence data base class.

    Used to build skill-based (skill/concept) knowledge tracing model data.
    """

    _sequence_length_key = "max_skill_seq_len"

    def __init__(self, data_src: DataSource, cache: bool = False):
        """Initialise the skill-level model data object.

        Args:
            data_src: Data source object.
            cache: Whether to enable disk caching.
        """
        super().__init__(data_src, cache=cache)
        self._split_getter = data_src.get_split_skill_sequence_data

    def load_windowlate_data(
        self,
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

        Returns:
            Tuple of (user_sequence, user_response, user_mask,
            late_group_id, user_true_labels, user_question,
            user_id_sequence) as numpy arrays,
            each of shape (num_samples, max_windowlate_seq_len) -- the
            maximum window row count recorded in metadata by preprocessing.
            user_id_sequence (original student ids) is last, matching
            WindowlateIterableDataset.
        """
        import numpy as np

        data = self.data_src.get_windowlate_data()

        if data is None:
            raise ValueError(
                "No windowlate data available. Please re-run preprocessing with K-fold labels."
            )
        max_seq_len = self.data_src.get_metadata("max_windowlate_seq_len")

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

    def create_windowlate_dataloader(
        self, *, batch_size: int, **kwargs: Any
    ) -> DataLoader:
        """Create the windowlate evaluation loader.

        Args:
            batch_size: Number of windows per batch.
            **kwargs: Worker, prefetch and other DataLoader options.

        Returns:
            A loader using the model's windowlate dataset and batch collator.
        """
        from utils.config.data_config import _create_dataloader

        return _create_dataloader(
            self._create_windowlate_dataset(), batch_size=batch_size, **kwargs
        )

    def _create_windowlate_dataset(self) -> WindowlateIterableDataset:
        """Create a WindowlateIterableDataset from the windowlate parquet file.

        Subclasses override this hook to supply model-specific window features.
        The padded length is the dataset's maximum window row count
        (metadata ``max_windowlate_seq_len``, recorded by preprocessing).

        Returns:
            A configured WindowlateIterableDataset instance.
        """
        parquet_path = os.path.join(
            self.data_src.data_folder, f"{self.data_src.dataset}_windowlate.parquet"
        )

        return WindowlateIterableDataset(
            parquet_path=parquet_path,
            max_seq_len=self.data_src.get_metadata("max_windowlate_seq_len"),
        )
