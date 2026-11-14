"""Polars-free dataset on-disk state inspection.

Encodes the data-layout convention (``<base>/<dataset>/raw/`` and
``metadata.json``) once so that both the web backend — which runs in a
polars-less environment and therefore cannot import :mod:`utils.data_process`
— and :class:`utils.data_process.DataSource` share a single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from utils.core import get_supported_datasets

# Written exclusively by ``DataSource.save_data`` (i.e. the ``process`` step);
# ``download`` writes metadata.json too but never this key, so its presence is
# the unambiguous "processed and ready for training" marker.
PROCESSED_MARKER = "sequence_data_md5"

DatasetStatus = Literal["empty", "downloaded", "ready"]


def data_folder_path(data_base_path: str | Path, dataset: str) -> Path:
    """Resolve ``<base>/<dataset>/`` (lowercased name, matching DataSource)."""
    return Path(data_base_path) / dataset.lower()


def raw_folder_path(data_base_path: str | Path, dataset: str) -> Path:
    """Resolve the extracted raw-data directory."""
    return data_folder_path(data_base_path, dataset) / "raw"


def metadata_path_for(data_base_path: str | Path, dataset: str) -> Path:
    """Resolve the dataset metadata.json path."""
    return data_folder_path(data_base_path, dataset) / "metadata.json"


def has_raw(data_base_path: str | Path, dataset: str) -> bool:
    """True if the raw-data directory exists and is non-empty."""
    raw_dir = raw_folder_path(data_base_path, dataset)
    return raw_dir.is_dir() and any(raw_dir.iterdir())


def is_processed(data_base_path: str | Path, dataset: str) -> bool:
    """True if metadata.json carries the processed marker."""
    meta_path = metadata_path_for(data_base_path, dataset)
    if not meta_path.exists():
        return False
    try:
        return PROCESSED_MARKER in json.loads(meta_path.read_text())
    except (ValueError, OSError):
        return False


def dataset_status(name: str, data_base_path: str | Path = "./data") -> DatasetStatus:
    """Classify a dataset as ``ready`` > ``downloaded`` > ``empty``.

    ``ready`` short-circuits before the raw check: a processed dataset may have
    had its raw files removed to save space, yet remains usable for training.
    """
    if is_processed(data_base_path, name):
        return "ready"
    if has_raw(data_base_path, name):
        return "downloaded"
    return "empty"


def list_dataset_statuses(
    data_base_path: str | Path = "./data",
) -> list[dict[str, str]]:
    """Status of every registered dataset, keyed by name."""
    return [
        {"name": name, "status": dataset_status(name, data_base_path)}
        for name in get_supported_datasets()
    ]
