"""Datasets router — listing and metadata retrieval.

Provides endpoints to list all available datasets and retrieve metadata
for a specific dataset by name.
"""

import json
from typing import Any

from config import PROJECT_ROOT
from errors import AppError
from fastapi import APIRouter
from services.registry_sync import registry_lock

from utils.core import get_supported_datasets
from utils.dataset_status import dataset_status

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATA_DIR = PROJECT_ROOT / "data"


@router.get("")
def list_datasets() -> Any:
    """List all available datasets with optional metadata summaries.

    Returns:
        A list of dicts, each containing ``name``, ``num_users``,
        ``num_questions``, and ``num_skills`` from the dataset's
        ``metadata.json`` if available.
    """
    with registry_lock:
        supported = get_supported_datasets()
    if not supported:
        raise AppError("no_datasets_registered", 500)

    result = []
    for name in supported:
        meta_path = DATA_DIR / name / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        else:
            meta = {}
        result.append(
            {
                "name": name,
                "status": dataset_status(name, DATA_DIR),
                "num_users": meta.get("num_users"),
                "num_questions": meta.get("num_questions"),
                "num_skills": meta.get("num_skills"),
            }
        )
    return result


@router.get("/{dataset_name}/metadata")
def get_dataset_metadata(dataset_name: str) -> Any:
    """Return the metadata JSON for a specific dataset.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        The parsed metadata dictionary.

    Raises:
        AppError: 404 if the dataset is unknown or its metadata file is
            absent, 500 if it cannot be read.
    """
    # Whitelist the path segment so a decoded "../../x" cannot traverse out of
    # DATA_DIR.
    with registry_lock:
        supported = get_supported_datasets()
    if dataset_name not in supported:
        raise AppError("dataset_not_found", 404)
    meta_path = DATA_DIR / dataset_name / "metadata.json"
    if not meta_path.exists():
        raise AppError("dataset_not_found", 404)
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        raise AppError("dataset_metadata_failed", 500)
