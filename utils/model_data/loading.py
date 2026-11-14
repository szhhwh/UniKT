"""DataLoader factories shared by training and windowlate evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from utils.config import DataLoaderConfig
from utils.core import get_logger

from .windowlate import WindowlateIterableDataset, _WindowlateBatchDataset

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset

logger = get_logger(__name__)


def _create_dataloader(dataset: Dataset[Any], **kwargs: Any) -> DataLoader[Any]:
    """Apply worker options and windowlate batching before constructing a loader."""
    # Prefetch and persistent workers only apply to multiprocessing loaders.
    if kwargs.get("num_workers", 0) == 0:
        kwargs.pop("prefetch_factor", None)
        kwargs["persistent_workers"] = False

    if isinstance(dataset, WindowlateIterableDataset):
        kwargs["collate_fn"] = dataset._collate_batch
        dataset = _WindowlateBatchDataset(dataset)
    return DataLoader(dataset, **kwargs)


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
        >>> from utils.config import DataLoaderConfig
        >>> from utils.model_data import create_optimized_dataloader
        >>> config = DataLoaderConfig(num_workers=4, pin_memory=True)
        >>> loader = create_optimized_dataloader(
        ...     dataset,
        ...     batch_size=64,
        ...     shuffle=True,
        ...     config=config,
        ...     device=torch.device("cuda")
        ... )
    """
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

    # Create DataLoader
    loader = _create_dataloader(dataset, **loader_kwargs)

    # Log optimization info
    logger.debug(
        f"Created optimized DataLoader: num_workers={loader.num_workers}, "
        f"pin_memory={loader.pin_memory}, "
        f"prefetch_factor={loader.prefetch_factor}, "
        f"persistent_workers={loader.persistent_workers}"
    )

    return loader


__all__ = ["create_optimized_dataloader"]
