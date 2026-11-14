"""Skill-level model data preparation and windowlate loader entry point."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .base_model_data import BaseModelData
from .loading import _create_dataloader
from .windowlate import WindowlateIterableDataset

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from utils.data_process import DataSource


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
