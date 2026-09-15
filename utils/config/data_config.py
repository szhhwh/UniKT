"""DataLoader configuration and the optimized DataLoader factory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..core import get_logger

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader, Dataset

logger = get_logger(__name__)

# Type definition for number of worker processes
NumWorkersType = int | Literal["auto"]


@dataclass
class DataLoaderConfig:
    """DataLoader configuration class.

    Attributes:
        num_workers: Number of worker processes for data loading.
                     "auto" sets it to min(CPU count, 8).
                     0 disables multiprocessing.
        pin_memory: Whether to pin tensors in CUDA memory (only effective on CUDA).
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


def create_optimized_dataloader(
    dataset: Dataset[Any],
    batch_size: int = 128,
    shuffle: bool = True,
    config: DataLoaderConfig | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    **kwargs: Any,
) -> DataLoader[Any]:
    """Create an optimized DataLoader.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        config: DataLoader configuration (defaults to DataLoaderConfig())
        device: Compute device (used to determine pin_memory)
        pin_memory: Host-memory pinning override (None = auto from
            config and device; never true on non-CUDA devices)
        **kwargs: Additional arguments passed to DataLoader (overrides config)

    Returns:
        An optimized DataLoader

    Example:
        >>> from utils.config import DataLoaderConfig, create_optimized_dataloader
        >>> config = DataLoaderConfig(num_workers=4, pin_memory=True)
        >>> loader = create_optimized_dataloader(
        ...     dataset,
        ...     batch_size=64,
        ...     shuffle=True,
        ...     config=config,
        ...     device=torch.device("cuda")
        ... )
    """
    from torch.utils.data import DataLoader

    # Use default configuration
    if config is None:
        config = DataLoaderConfig()

    # Device information is required
    if device is None:
        raise ValueError(
            "Device information is required to determine pin_memory setting."
        )

    # Determine pin_memory
    is_cuda = device.type == "cuda"
    if pin_memory is None:
        pin_memory = config.pin_memory
    pin_memory = pin_memory and is_cuda

    # Get num_workers
    num_workers = config.get_num_workers()

    # Prepare DataLoader arguments; kwargs take priority over config
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": config.prefetch_factor if num_workers > 0 else None,
        "persistent_workers": config.persistent_workers if num_workers > 0 else False,
    }

    # Override default arguments with kwargs
    loader_kwargs.update(kwargs)

    # prefetch/persistent require multiprocessing; normalize after overrides
    if loader_kwargs["num_workers"] == 0:
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs["persistent_workers"] = False

    # Create DataLoader
    loader = DataLoader(dataset, **loader_kwargs)

    # Log optimization info
    logger.debug(
        f"Created optimized DataLoader: num_workers={loader_kwargs.get('num_workers')}, "
        f"pin_memory={loader_kwargs.get('pin_memory')}, "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'N/A')}, "
        f"persistent_workers={loader_kwargs.get('persistent_workers', 'N/A')}"
    )

    return loader


__all__ = [
    "DataLoaderConfig",
    "create_optimized_dataloader",
]
