"""Data processing package.

Importing this package triggers static registration discovery (scans source files
without importing them), writing all ``@register_data_source`` entries into the
``DATA_SOURCES`` lazy index. Data source code is imported on demand only when
``get_data_source(...)`` is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from utils.config import GeneralConfig, RunDataConfig
from utils.core import DATA_SOURCES, discover_registrations, get_supported_datasets

from .data_source import DataSource

if TYPE_CHECKING:
    import argparse


class _HasRCNodes(Protocol):
    """Structural rc contract: only the ``data`` and ``general`` nodes are consumed.

    Satisfied by a full ``RunConfig`` (train.py) and by data_process.py's
    ``_PartialRC`` (the download/process CLI collects no model node).
    """

    data: RunDataConfig
    general: GeneralConfig


discover_registrations(Path(__file__).parent, "utils.data_process")


def _rc_to_args_namespace(rc: _HasRCNodes) -> argparse.Namespace:
    """Flatten a RunConfig's data + general nodes into the flat namespace DataSource consumes.

    Yields ``args.dataset`` / ``args.seed`` / ``args.min_seq_len`` / ... for
    DataSource internals.
    """
    import argparse

    from utils.config import config_to_dict

    flat = config_to_dict(rc.data)
    flat["seed"] = rc.general.seed
    flat["device"] = rc.general.device
    return argparse.Namespace(**flat)


def get_data_source(rc: _HasRCNodes) -> DataSource:
    """Get a data source instance from a RunConfig, with on-demand lazy import.

    Args:
        rc: RunConfig instance; the dataset is read from
            ``rc.data.dataset``.

    Returns:
        A configured ``DataSource`` instance.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    dataset_name = rc.data.dataset
    if dataset_name not in DATA_SOURCES:
        available = ", ".join(get_supported_datasets())
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {available}")
    dataset_cls = DATA_SOURCES.get(dataset_name)
    return dataset_cls(args=_rc_to_args_namespace(rc))


__all__ = ["DataSource", "get_data_source"]
