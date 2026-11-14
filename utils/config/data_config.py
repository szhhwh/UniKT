"""DataLoader worker and host-memory configuration."""

import os
from dataclasses import dataclass
from typing import Literal

# Type definition for number of worker processes
NumWorkersType = int | Literal["auto"]


@dataclass
class DataLoaderConfig:
    """DataLoader configuration class.

    Attributes:
        num_workers: Number of worker processes for data loading.
                     "auto" sets it to min(CPU count, 8).
                     0 disables multiprocessing.
        pin_memory: Whether to pin host memory for CUDA transfers.
        prefetch_factor: Number of batches to prefetch per worker (only effective when num_workers > 0).
        persistent_workers: Whether to keep workers alive between epochs (PyTorch >= 1.7).
    """

    num_workers: NumWorkersType = "auto"
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    def get_num_workers(self, max_limit: int = 8) -> int:
        """Get the actual num_workers value.

        Args:
            max_limit: Maximum number of worker processes

        Returns:
            The actual num_workers value
        """
        if self.num_workers == "auto":
            cpu_count = os.cpu_count() or 1
            return min(cpu_count, max_limit)
        return self.num_workers


__all__ = ["DataLoaderConfig"]
