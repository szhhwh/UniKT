"""Default case data sink accumulating results into a pandas DataFrame.

Defines the canonical parquet schema consumed by the default selectors
and visualizers: one row per attempt, columns ``user_id``,
``question_id``, ``label``, ``prediction`` (required), plus ``skill``,
``logit``, ``mask``, ``knowledge_state`` and any extra keys the
analyzer emits (passed through under their original names). The
``position`` column (per-user sequence order) is always derived.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from ...core import get_logger, register_case_sink
from ..interfaces import CaseDataSink

logger = get_logger(__name__)

_REQUIRED_KEYS = ("user_ids", "question_ids", "labels", "predictions")

_CANONICAL = {
    "user_ids": "user_id",
    "question_ids": "question_id",
    "skills": "skill",
    "labels": "label",
    "predictions": "prediction",
    "logits": "logit",
    "mask": "mask",
    "knowledge_states": "knowledge_state",
}

_RESULT_REQUIRED_COLUMNS = ("user_id", "label", "prediction")


@register_case_sink("dataframe")
class DataFrameSink(CaseDataSink):
    """Accumulates case data batches into a DataFrame and persists parquet."""

    def __init__(self) -> None:
        """Initialize an empty sink with no cached DataFrame."""
        self._columns: dict[str, list] | None = None
        self._df: pd.DataFrame | None = None

    def add_batch(self, case_data: dict[str, Any]) -> None:
        """Accumulate one batch of extracted case data.

        Args:
            case_data: Parallel lists keyed by plural batch names
                (``user_ids`` etc.). Keys must stay identical across
                batches; unknown keys are passed through as extra
                columns under their original names.

        Raises:
            ValueError: If a required key is missing, batch lengths
                differ, or the key set changed between batches.
        """
        missing = [k for k in _REQUIRED_KEYS if k not in case_data]
        if missing:
            raise ValueError(f"Missing required keys in case_data: {missing}")

        batch = {
            _CANONICAL.get(key, key): list(value) for key, value in case_data.items()
        }
        lengths = {c: len(v) for c, v in batch.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Inconsistent data lengths in batch: {lengths}. "
                "All keys must have the same number of samples."
            )

        if self._columns is None:
            self._columns = {c: [] for c in batch}
        elif set(batch) != set(self._columns):
            raise ValueError(
                "Inconsistent case_data keys across batches: "
                f"got {sorted(batch)}, expected {sorted(self._columns)}"
            )

        for column, values in batch.items():
            self._columns[column].extend(values)
        self._df = None

    def result(self) -> pd.DataFrame:
        """Return the accumulated DataFrame with derived ``position``."""
        if self._df is None:
            if self._columns is None or not self._columns:
                return pd.DataFrame()
            df = pd.DataFrame(self._columns)
            # Insertion order doubles as per-user sequence order
            df["position"] = df.groupby("user_id").cumcount()
            self._df = df
        return self._df

    @staticmethod
    def save(df: pd.DataFrame, output_path: str) -> None:
        """Save a results DataFrame to a parquet file.

        Args:
            df: Results DataFrame (typically ``DataFrameSink.result()``).
            output_path: Destination parquet path.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


def load_case_results(path: str) -> pd.DataFrame:
    """Load canonical case results from a parquet file.

    Args:
        path: Parquet path written by :meth:`DataFrameSink.save`.

    Returns:
        DataFrame with ``position`` re-derived when absent.

    Raises:
        ValueError: If required canonical columns are missing.
    """
    logger.info(f"Loading results from {path}...")
    df = pd.read_parquet(path)
    missing = [c for c in _RESULT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if "position" not in df.columns:
        df["position"] = df.groupby("user_id").cumcount()
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def get_user_sequence(df: pd.DataFrame, user_id: Any) -> pd.DataFrame:
    """Get one user's full sequence sorted by position.

    Args:
        df: Canonical case results DataFrame.
        user_id: User identifier.

    Returns:
        The user's rows sorted by ``position`` with a fresh index.
    """
    user_df = df[df["user_id"] == user_id].copy()
    return user_df.sort_values("position").reset_index(drop=True)


__all__ = ["DataFrameSink", "get_user_sequence", "load_case_results"]
