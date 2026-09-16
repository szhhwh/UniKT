"""Data source management for knowledge tracing datasets.

Provides the base DataSource class and utility functions for downloading,
loading, processing, and saving dataset files with metadata tracking and
integrity checks.
"""

import hashlib
import json
import os
import shutil
import time
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, overload

import numpy as np
import polars as pl
import requests
import tqdm
from sklearn.model_selection import KFold

from utils.core import get_logger
from utils.dataset_status import has_raw, is_processed, raw_folder_path

from .windowlate_processor import WindowlateProcessor

logger = get_logger(__name__)


class DataSource(ABC):
    """Base class for data source management.

    Provides common functionality for downloading, loading, processing,
    and saving dataset files with metadata tracking and integrity checks.
    """

    # Injected by dataset-specific subclasses (Assistments2009Data, ...) —
    # a processed Namespace carrying min_seq_len/max_seq_len etc.
    args: Any
    # The processed user sequence table, assigned during the subclass
    # process flow; always an eager DataFrame (lazy policy is windowlate-only).
    sequence_data: pl.DataFrame
    # Raw per-interaction tables populated by subclass load_src_data /
    # clean_raw_data; None until those steps run. Structure is subclass-
    # defined (LazyFrame / eager DataFrame / per-source dict), hence Any.
    raw_data: Any
    cleaned_raw_data: Any

    def __init__(
        self,
        dataset: str,
        data_base_path: str,
        data_url: str | None = None,
        seed: int = 42,
    ):
        """Initialize the DataSource with dataset configuration."""
        super().__init__()
        self.dataset = dataset.lower()
        self.data_base_path = data_base_path
        self.data_folder = os.path.join(self.data_base_path, self.dataset)
        self.metadata_path = os.path.join(self.data_folder, "metadata.json")
        self.raw_data = None
        self.cleaned_raw_data = None  # cleaned raw data
        self.data_url = data_url
        self.metadata: dict[str, Any] = {}
        self.seed = seed

        # ID mapping storage
        self._id_mappings: dict = {}

        # Normalized relation tables: each key is "question_{entity}"
        # e.g., "question_skill", "question_assignment", "question_template"
        self.relation_data: dict[str, pl.DataFrame] = {}

        # Data cache
        self._data_cache: dict[str, pl.DataFrame | pl.LazyFrame] = {}
        self._data_config: dict[str, dict] = {
            "sequence": {"lazy": False},
            "split_question_sequence": {"lazy": False},
            "split_skill_sequence": {"lazy": False},
            "windowlate": {"lazy": True},
        }

    @property
    def raw_folder(self) -> str:
        """Extracted raw-data directory (``<data_folder>/raw``)."""
        return str(raw_folder_path(self.data_base_path, self.dataset))

    def raw_exists(self) -> bool:
        """True if the raw-data directory exists and is non-empty."""
        return has_raw(self.data_base_path, self.dataset)

    def metadata_exists(self) -> bool:
        """True if ``metadata.json`` is present on disk."""
        return os.path.exists(self.metadata_path)

    def processed_exists(self) -> bool:
        """True if the dataset has been processed (metadata carries the marker).

        Distinct from :meth:`metadata_exists`: ``download`` also writes
        metadata.json, but only ``process`` records the processed marker.
        """
        return is_processed(self.data_base_path, self.dataset)

    def _build_id_mapping(self, data: pl.DataFrame, columns: list[str]) -> None:
        """Build ID mappings from data.

        Args:
            data: Polars DataFrame
            columns: List of column names to build mappings for
        """
        for col in columns:
            if col in self._id_mappings:
                logger.warning(
                    f"ID mapping for column '{col}' already exists, skipping rebuild"
                )
                continue  # Already have mapping, skip

            unique_vals = data[col].unique().sort().to_list()
            value_map = {val: idx for idx, val in enumerate(unique_vals)}

            self._id_mappings[col] = value_map
            logger.debug(
                f"Built ID mapping for '{col}': {len(value_map)} unique values"
            )

    def _apply_id_mapping(self, data: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        """Apply ID mappings to data.

        Args:
            data: Polars DataFrame
            columns: List of column names to map

        Returns:
            DataFrame with mapped columns
        """
        for col in columns:
            if col not in self._id_mappings:
                raise ValueError(f"No ID mapping exists for column '{col}'")

            value_map = self._id_mappings[col]
            data = data.with_columns(
                pl.col(col).replace(value_map).cast(pl.Int32).alias(col)
            )

        return data

    def _get_mapped_count(self, column: str) -> int:
        """Get the count of mapped IDs for a column."""
        if column not in self._id_mappings:
            raise ValueError(f"No ID mapping exists for column '{column}'")
        return len(self._id_mappings[column])

    def _export_id_mappings(self) -> dict:
        """Export ID mappings as a dictionary."""
        return {
            f"id_mapping_{col}": mapping for col, mapping in self._id_mappings.items()
        }

    def _sort_columns(self, primary: list[str] | None = None) -> list[str]:
        """Get column sort order with primary keys first, remaining columns as tie-breakers."""
        if primary is None:
            primary = ["user", "timestamp"]
        return primary + [c for c in self.sequence_data.columns if c not in primary]

    def save_data(self) -> None:
        """Save processed relation tables, sequence data, and metadata."""
        # Validate
        self._validate_data(self.relation_data, self.sequence_data)

        # Save relation tables with deterministic row order
        relation_md5s = {}
        for name, df in self.relation_data.items():
            path = os.path.join(
                self.data_folder, f"{self.dataset}_relation_{name}.parquet"
            )
            df.sort(df.columns).write_parquet(path)
            relation_md5s[f"relation_{name}_md5"] = self.compute_md5(path)
            logger.debug(f"Saved relation {name} to: {path}")

        # Save sequence data
        sequence_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_sequence.parquet"
        )
        split_question_sequence_path = os.path.join(
            self.data_folder, f"{self.dataset}_split_question_sequence.parquet"
        )
        split_skill_sequence_path = os.path.join(
            self.data_folder, f"{self.dataset}_split_skill_sequence.parquet"
        )
        windowlate_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_windowlate.parquet"
        )

        self.sequence_data.sort(self._sort_columns()).write_parquet(sequence_data_path)
        self.split_question_sequence_data.write_parquet(split_question_sequence_path)
        self.split_skill_sequence_data.write_parquet(split_skill_sequence_path)

        # Save metadata
        metadata = {
            "min_seq_len": self.args.min_seq_len,
            "max_seq_len": self.args.max_seq_len,
            "random_seed": self.seed,
            "sequence_data_md5": self.compute_md5(sequence_data_path),
            "split_question_sequence_data_md5": self.compute_md5(
                split_question_sequence_path
            ),
            "split_skill_sequence_data_md5": self.compute_md5(
                split_skill_sequence_path
            ),
            "num_users": self.sequence_data["user"].n_unique(),
            "num_questions": self.sequence_data["question"].n_unique(),
            "num_skills": self.relation_data["question_skill"]["skill"].n_unique(),
            "num_split_question_users": self.split_question_sequence_data[
                "sequence_id"
            ].n_unique(),
            "num_split_skill_users": self.split_skill_sequence_data[
                "sequence_id"
            ].n_unique(),
        }

        # Optional entity counts from their relation tables
        if "question_assignment" in self.relation_data:
            metadata["num_assignments"] = self.relation_data["question_assignment"][
                "assignment"
            ].n_unique()
        if "question_template" in self.relation_data:
            metadata["num_templates"] = self.relation_data["question_template"][
                "template"
            ].n_unique()

        metadata.update(relation_md5s)

        logger.debug(f"Saved sequence_data to: {sequence_data_path}")
        logger.debug(
            f"Saved split question sequences to: {split_question_sequence_path}"
        )
        logger.debug(f"Saved split skill sequences to: {split_skill_sequence_path}")
        if os.path.exists(windowlate_data_path):
            metadata["windowlate_data_md5"] = self.compute_md5(windowlate_data_path)
            logger.debug(f"Windowlate data already saved to: {windowlate_data_path}")
        logger.info(f"Data saved to {self.data_folder}")

        self.update_metadatas(metadata)
        self.save_metadata()

    @staticmethod
    def _validate_data(
        relation_data: dict[str, pl.DataFrame], sequence_data: pl.DataFrame
    ) -> None:
        """Validate consistency between relation tables and sequence_data."""
        assert "question_skill" in relation_data, "question_skill relation is required"
        question_skill = relation_data["question_skill"]

        # Validate question_skill columns
        assert set(question_skill.columns) == {"question", "skill"}, (
            f"question_skill columns mismatch. "
            f"Expected: {{question, skill}}, Got: {set(question_skill.columns)}"
        )

        # Validate each relation has exactly 2 columns and is unique
        for name, df in relation_data.items():
            assert len(df.columns) == 2, (
                f"Relation '{name}' should have exactly 2 columns, got {df.columns}"
            )
            assert df.unique(subset=df.columns).shape[0] == df.shape[0], (
                f"Relation '{name}' has duplicate rows"
            )

        # Validate sequence_data doesn't contain skill
        assert "skill" not in sequence_data.columns, (
            "sequence_data should not contain 'skill' column"
        )

        # Validate question_id consistency
        q_questions = set(question_skill["question"].unique().to_list())
        s_questions = set(sequence_data["question"].unique().to_list())

        missing_questions = s_questions - q_questions
        if missing_questions:
            raise AssertionError(
                f"question_id mismatch: {len(missing_questions)} questions in sequence_data "
                f"not found in question_skill relation"
            )

        assert q_questions == s_questions, (
            f"question_id mismatch: question_skill has {len(q_questions)} unique, "
            f"sequence_data has {len(s_questions)} unique"
        )

        # Validate ID range consistency (Any: polars .max() unions trigger
        # str-bytes-safe false positives on the f-string below)
        q_max: Any = question_skill["question"].max()
        s_max: Any = sequence_data["question"].max()
        assert q_max == s_max, (
            f"question_id range mismatch: question_skill max={q_max}, "
            f"sequence_data max={s_max}"
        )

        logger.info("Data validation passed!")

    def _download_chunk(
        self,
        url: str | None,
        start: int,
        end: int,
        chunk_path: str,
        pbar: tqdm.tqdm | None,
        chunk_size: int = 8192,
    ) -> None:
        """Download a single chunk of a file for multi-threaded downloads."""
        headers = {"Range": f"bytes={start}-{end}"}
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with open(chunk_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if pbar:
                        pbar.update(len(chunk) / (1024 * 1024))

    def _download_with_requests(
        self, archive_path: str, num_threads: int, attempt: int, max_retries: int
    ) -> None:
        """Download file with multi-threading support."""
        head_response = requests.head(self.data_url, timeout=30)
        head_response.raise_for_status()

        total_bytes = int(head_response.headers.get("content-length", 0))
        accept_ranges = head_response.headers.get("accept-ranges", "none")

        if accept_ranges != "bytes" or total_bytes < 10 * 1024 * 1024:
            if accept_ranges != "bytes":
                logger.warning(
                    "Server does not support range requests, using single-threaded download"
                )
            self._download_single_thread(archive_path)
            return

        total_mb = total_bytes / (1024 * 1024)
        chunk_size = total_bytes // num_threads
        temp_dir = archive_path + ".parts"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            with tqdm.tqdm(
                total=total_mb,
                unit="MB",
                unit_scale=False,
                desc=f"Downloading (attempt {attempt}/{max_retries})",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}MB [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = []
                    for i in range(num_threads):
                        start = i * chunk_size
                        end = (
                            start + chunk_size - 1
                            if i < num_threads - 1
                            else total_bytes - 1
                        )
                        chunk_path = os.path.join(temp_dir, f"chunk_{i}")

                        future = executor.submit(
                            self._download_chunk,
                            self.data_url,
                            start,
                            end,
                            chunk_path,
                            pbar,
                        )
                        futures.append(future)

                    for future in futures:
                        future.result()

            logger.debug("Merging downloaded chunks...")
            with open(archive_path, "wb") as outfile:
                for i in range(num_threads):
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}")
                    with open(chunk_path, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _download_single_thread(self, archive_path: str) -> None:
        """Download file with single thread."""
        with requests.get(self.data_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            total_mb = total_bytes / (1024 * 1024)
            chunk_size = 8192

            with open(archive_path, "wb") as f:
                with tqdm.tqdm(
                    total=total_mb,
                    unit="MB",
                    unit_scale=False,
                    desc="Downloading",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}MB [{elapsed}<{remaining}, {rate_fmt}]",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk) / (1024 * 1024))

    def fetch_data(
        self, force_download: bool = False, max_retries: int = 3, num_threads: int = 4
    ) -> None:
        """Download and extract data archive with retry support.

        Args:
            force_download: Force re-download even if file exists.
            max_retries: Maximum number of download retry attempts.
            num_threads: Number of threads for multi-threaded download.
        """
        if self.data_url is None:
            raise ValueError("Data URL is not provided.")

        os.makedirs(self.data_folder, exist_ok=True)

        file_name = self.data_url.split("/")[-1]
        if not file_name:
            file_name = "downloaded_data"
        archive_path = os.path.join(self.data_folder, file_name)

        if not os.path.exists(archive_path) or force_download:
            if force_download and os.path.exists(archive_path):
                logger.warning(
                    f"Force download enabled, removing existing file: {archive_path}"
                )
                os.remove(archive_path)

            logger.info(f"Downloading data from {self.data_url}")

            for attempt in range(max_retries):
                try:
                    self._download_with_requests(
                        archive_path, num_threads, attempt + 1, max_retries
                    )
                    logger.info(f"Download finished: {archive_path}")
                    break
                except Exception as e:
                    if os.path.exists(archive_path):
                        os.remove(archive_path)
                        logger.debug(f"Removed incomplete download: {archive_path}")

                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.error(
                            f"Download failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Failed to download data after {max_retries} attempts: {e}"
                        )
        else:
            logger.info(f"Dataset already exists, skip downloading: {archive_path}")

        archive_md5 = self.compute_md5(archive_path)
        self.update_metadata("raw_archive_md5", archive_md5)
        self.update_metadata("raw_archive_filename", file_name)

        extract_target = os.path.join(self.data_folder, "raw")
        os.makedirs(extract_target, exist_ok=True)

        should_extract = self._should_extract(force_download, extract_target)

        if should_extract:
            logger.info(f"Extracting archive: {archive_path}")
            self._extract_archive(archive_path, file_name, extract_target)
            logger.info(f"Extraction finished: {extract_target}")
        else:
            logger.info(
                f"Raw data directory not empty, skip extraction: {extract_target}"
            )

        self.update_metadata("raw_data_path", extract_target)

    def _should_extract(self, force_download: bool, extract_target: str) -> bool:
        """Determine whether extraction is needed."""
        if force_download:
            if any(Path(extract_target).iterdir()):
                logger.warning(
                    f"Force mode enabled, removing existing raw data: {extract_target}"
                )
                shutil.rmtree(extract_target)
                os.makedirs(extract_target, exist_ok=True)
            return True
        return not any(Path(extract_target).iterdir())

    def _extract_archive(
        self, archive_path: str, file_name: str, extract_target: str
    ) -> None:
        """Extract archive file based on its extension."""
        import gzip
        import tarfile

        lower_name = file_name.lower()

        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                # zipfile has no extraction filter; reject members whose
                # resolved path escapes the destination.
                dest_root = os.path.realpath(extract_target)
                for member in zf.namelist():
                    member_path = os.path.realpath(os.path.join(extract_target, member))
                    if not member_path.startswith(dest_root + os.sep):
                        raise ValueError(
                            f"Refusing to extract archive member outside "
                            f"destination: {member}"
                        )
                zf.extractall(extract_target)
        elif lower_name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                # "data" filter (PEP 706): archive contents are untrusted, so
                # absolute paths, ".." traversal, and special files are rejected.
                tf.extractall(extract_target, filter="data")
        elif lower_name.endswith(".tar"):
            with tarfile.open(archive_path, "r:") as tf:
                tf.extractall(extract_target, filter="data")
        elif lower_name.endswith(".gz") and not lower_name.endswith(".tar.gz"):
            uncompressed_name = lower_name[:-3]
            target_file = os.path.join(extract_target, uncompressed_name)
            with (
                gzip.open(archive_path, "rb") as f_in,
                open(target_file, "wb") as f_out,
            ):
                shutil.copyfileobj(f_in, f_out)
        else:
            dest_path = os.path.join(extract_target, file_name)
            if archive_path != dest_path:
                shutil.copy2(archive_path, dest_path)

    @abstractmethod
    def load_src_data(self) -> None:
        """Load source data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement load_data method")

    def _validate_saved_data(self, data_name: str) -> str:
        """Validate that processed data files exist and have correct integrity."""
        # Construct the file path
        data_path = os.path.join(
            self.data_folder, f"{self.dataset}_{data_name}.parquet"
        )
        # Check if the file exists
        self._validate_data_files_exist([data_path])
        # Validate file integrity
        self._validate_data_integrity(data_name, data_name + "_md5")

        return data_path

    def _validate_data_files_exist(self, file_paths: list[str]) -> None:
        """Validate that all required data files exist."""
        missing_files = [
            f"  - {path}" for path in file_paths if not os.path.exists(path)
        ]

        if missing_files:
            missing_str = "\n".join(missing_files)
            raise FileNotFoundError(
                f"Processed data files not found for dataset '{self.dataset}':\n"
                f"{missing_str}\n\n"
                f"To fix this, please run preprocessing first:\n"
                f"   python data_process.py process -d {self.dataset}\n\n"
                f"Data base path: {self.data_folder}"
            )

    def _validate_data_integrity(self, file_path: str, md5_key: str) -> None:
        """Validate MD5 checksum of a data file."""
        if md5_key not in self.metadata:
            return

        actual_md5 = self.compute_md5(file_path)
        expected_md5 = self.metadata[md5_key]

        if actual_md5 != expected_md5:
            data_type = (
                "relation"
                if "relation" in md5_key
                else "sequence"
                if "sequence" in md5_key
                else "split_sequence"
            )
            raise ValueError(
                f"{data_type.capitalize()} data file integrity check failed (MD5 mismatch).\n"
                f"Expected MD5: {expected_md5}\n"
                f"Actual MD5: {actual_md5}\n\n"
                f"The data may be corrupted or outdated. Please re-run preprocessing:\n"
                f"   python data_process.py process -d {self.dataset}"
            )

    @abstractmethod
    def transform_data(self) -> None:
        """Transform cleaned data into standard format. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement transform_data method")

    @abstractmethod
    def clean_raw_data(self) -> None:
        """Clean raw data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement clean_raw_data method")

    def compute_md5(self, file_path: str) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hash_md5.update(chunk)

        logger.debug(f"Computed MD5 for {file_path}: {hash_md5.hexdigest()}")

        return hash_md5.hexdigest()

    def _load_data(self, data_type: str) -> pl.DataFrame | pl.LazyFrame:
        """Load data by type using the configured lazy/eager policy.

        Args:
            data_type: Data type key corresponding to the configuration dict.

        Returns:
            DataFrame or LazyFrame.

        Raises:
            ValueError: If data_type is not in the configuration dict.
        """
        if data_type not in self._data_config:
            raise ValueError(f"Unknown data type: {data_type}")

        # Check cache
        if data_type in self._data_cache:
            return self._data_cache[data_type]

        config = self._data_config[data_type]
        data_path = self._validate_saved_data(data_type)

        # Read according to configuration (lazy vs eager)
        read_func = pl.scan_parquet if config["lazy"] else pl.read_parquet
        logger.info(f"Loading {data_type} data: {data_path}")
        data = read_func(data_path)

        # Cache the loaded data
        self._data_cache[data_type] = data

        return data

    def get_sequence_data(self) -> pl.DataFrame:
        """Get user sequence data."""
        data = self._load_data("sequence")
        assert isinstance(data, pl.DataFrame)  # sequence is always eager
        return data

    def get_relation(self, name: str) -> pl.DataFrame:
        """Get a normalized relation table by name (e.g., "question_skill").

        Each relation is a 2-column DataFrame unique on (src, dst).
        """
        if name in self._data_cache:
            cached = self._data_cache[name]
            assert isinstance(cached, pl.DataFrame)  # relations are eager
            return cached
        if self.relation_data and name in self.relation_data:
            return self.relation_data[name]
        # Try loading from disk
        path = os.path.join(self.data_folder, f"{self.dataset}_relation_{name}.parquet")
        if not os.path.exists(path):
            raise ValueError(
                f"Relation '{name}' not found at {path}. "
                f"Please re-run: python data_process.py process -d {self.dataset}"
            )
        logger.info(f"Loading relation {name} from: {path}")
        data = pl.read_parquet(path)
        self._data_cache[name] = data
        return data

    def get_available_relations(self) -> list[str]:
        """Return names of available relation tables."""
        if self.relation_data:
            return list(self.relation_data.keys())
        # Scan disk for relation files
        import glob

        pattern = os.path.join(self.data_folder, f"{self.dataset}_relation_*.parquet")
        paths = glob.glob(pattern)
        return [
            p.split("_relation_")[1].replace(".parquet", "")
            for p in paths
            if "_relation_" in p
        ]

    def get_split_question_sequence_data(self) -> pl.DataFrame:
        """Get split user sequence data."""
        data = self._load_data("split_question_sequence")
        assert isinstance(data, pl.DataFrame)  # split data is always eager
        return data

    def get_split_skill_sequence_data(self) -> pl.DataFrame:
        """Get split skill sequence data."""
        data = self._load_data("split_skill_sequence")
        assert isinstance(data, pl.DataFrame)  # split data is always eager
        return data

    def get_windowlate_data(self) -> pl.LazyFrame:
        """Get windowlate evaluation data.

        Returns:
            Windowlate evaluation samples (long format).
            Columns: sample_id, position, skill, response, mask, user_id, group_id, true_label, fold
        """
        data = self._load_data("windowlate")
        assert isinstance(data, pl.LazyFrame)  # windowlate is always lazy
        return data

    def update_metadata(self, key: str, value: Any) -> None:
        """Update a single metadata entry."""
        self.metadata[key] = value
        logger.debug(f"Updated {key} = {value} in DataSource metadata")

    def update_metadatas(self, meta_dict: dict[str, Any]) -> None:
        """Update multiple metadata entries."""
        for key, value in meta_dict.items():
            self.update_metadata(key, value)

    def save_metadata(self) -> None:
        """Save metadata to JSON file."""
        self.update_metadata("dataset", self.dataset)
        self.update_metadata("data_base_path", self.data_base_path)

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def load_metadata(self) -> None:
        """Load metadata from JSON file."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

    def get_metadata(self, key: str | None = None) -> Any:
        """Get metadata entry or entire metadata dict."""
        if not self.metadata:
            self.load_metadata()
        if key is not None and key not in self.metadata:
            raise KeyError(
                f"Metadata key '{key}' not found in dataset '{self.dataset}'"
            )
        return self.metadata if key is None else self.metadata[key]

    # Target number of source rows processed per batch when splitting
    # sequences. Batches are aligned to user boundaries so a single user is
    # never split across batches. This bounds peak memory regardless of how
    # large the (possibly skill-exploded) dataset grows, while producing output
    # that is identical to whole-frame processing.
    _SPLIT_BATCH_ROWS: int = 2_000_000

    def _iter_user_aligned_slices(
        self, df: pl.DataFrame, target_rows: int
    ) -> Iterator[tuple[int, int]]:
        """Yield (start, end) row offsets of contiguous, user-aligned batches.

        ``df`` must be sorted by the ``user`` column. Each yielded batch
        contains whole users (a user is never split across batches) and spans
        at least ``target_rows`` rows, except for the final batch. A single
        user longer than ``target_rows`` forms its own (oversized) batch.

        Yields:
            Tuples of (start_offset, end_offset) for use with ``df.slice``.
        """
        user_col = df["user"].to_numpy()
        n = user_col.size
        if n == 0:
            return
        # Indices where a new user group starts (data is already sorted by user).
        change = np.flatnonzero(user_col[1:] != user_col[:-1]) + 1
        boundaries = np.concatenate(([0], change, [n]))
        num_groups = boundaries.size - 1

        start = 0
        while start < num_groups:
            end = start + 1
            while end < num_groups and (
                boundaries[end] - boundaries[start] < target_rows
            ):
                end += 1
            yield int(boundaries[start]), int(boundaries[end])
            start = end

    def _build_split_sequences(self, expand_skills: bool) -> pl.DataFrame:
        """Build split sequences, processing users in memory-bounded batches.

        The split operation is independent per user, so users are processed in
        contiguous batches aligned to user boundaries. Peak memory is bounded
        by the batch size rather than the full (possibly skill-exploded)
        dataset. The output is identical to whole-frame processing: every per-
        user operation (sequence position, length, split index/length, dense
        new-user-id assignment in global ``(user, split_idx)`` order) is
        preserved because no user ever spans two batches.

        Args:
            expand_skills: If True, expand each question into its skills (one
                row per skill) before splitting, producing skill sequences. If
                False, split the question sequences unchanged.

        Returns:
            DataFrame with the original ``sequence_data`` columns (including
            the pre-split ``user`` student id) plus a ``sequence_id`` column
            (the dense split id assigned in global ``(user, split_idx)`` order)
            and a ``seq_pos`` column (and an extra ``skill`` column when
            ``expand_skills``).
        """
        max_seq_len = self.args.max_seq_len
        min_seq_len = self.args.min_seq_len
        seq_cols = list(self.sequence_data.columns)

        # Ensure chronological order within each user, breaking timestamp ties
        # with the remaining columns (matches the previous whole-frame sort).
        self.sequence_data = self.sequence_data.sort(self._sort_columns())

        question_skills = None
        if expand_skills:
            if not self.relation_data or "question_skill" not in self.relation_data:
                raise ValueError("question_skill relation not available.")
            # One row per question with its skills sorted ascending. Small
            # (~num_questions rows) and reused across every batch.
            question_skills = (
                self.relation_data["question_skill"]
                .sort("question", "skill")
                .group_by("question")
                .agg(pl.col("skill").sort().alias("skills"))
            )
            logger.info(
                f"Building split skill sequences (max_len={max_seq_len}, "
                f"min_len={min_seq_len})"
            )
        else:
            logger.info(
                f"Building split question sequences (max_len={max_seq_len}, "
                f"min_len={min_seq_len})"
            )

        parts: list[pl.DataFrame] = []
        next_new_id = 0
        for batch_idx, (start, end) in enumerate(
            self._iter_user_aligned_slices(self.sequence_data, self._SPLIT_BATCH_ROWS)
        ):
            batch = self.sequence_data.slice(start, end - start)

            if expand_skills:
                assert question_skills is not None  # set when expand_skills
                # The hash join does not guarantee row order, so we must
                # preserve the deterministic chronological (interaction-major)
                # order ourselves. Stamp each interaction with its position in
                # the (already sorted) batch before joining, then explode and
                # restore order by (position, skill): interactions stay in
                # chronological order, with each interaction's skills ascending.
                # This reproduces the previous whole-frame output exactly --
                # including users with repeated interactions (same question and
                # timestamp) -- and is independent of batch size.
                batch = batch.with_columns(pl.int_range(pl.len()).alias("__order"))
                batch = (
                    batch.join(question_skills, on="question", how="inner")
                    .explode("skills")
                    .rename({"skills": "skill"})
                    .sort(["__order", "skill"])
                    .drop("__order")
                )

            # Per-user sequence position (0..L-1) and total length L.
            b = batch.with_columns(pl.int_range(pl.len()).over("user").alias("seq_pos"))
            b = b.join(
                b.group_by("user").agg(pl.len().alias("seq_len")),
                on="user",
                how="left",
            )

            # Split index and per-split length (constant within a split).
            b = b.with_columns(
                (pl.col("seq_pos") // max_seq_len).alias("split_idx")
            ).with_columns(
                pl.when(pl.col("seq_pos") + max_seq_len >= pl.col("seq_len"))
                .then(pl.col("seq_len") - pl.col("split_idx") * max_seq_len)
                .otherwise(max_seq_len)
                .alias("split_len"),
            )

            # Keep only splits long enough, then assign dense new user ids.
            valid = (
                b.filter(pl.col("split_len") >= min_seq_len)
                .select(["user", "split_idx"])
                .unique()
                .sort("user", "split_idx")
                .with_row_index("new_user_id")
            )
            n_new = valid.height
            if n_new == 0:
                continue
            # Offset the locally-assigned ids by the running global counter so
            # the global (user, split_idx) ordering is preserved across batches.
            valid = valid.with_columns(
                (pl.col("new_user_id").cast(pl.Int64) + next_new_id)
                .cast(pl.Int32)
                .alias("new_user_id")
            )

            b = b.join(valid, on=["user", "split_idx"], how="inner")
            b = b.with_columns(
                [
                    pl.col("new_user_id").alias("sequence_id"),
                    (pl.col("seq_pos") % max_seq_len).alias("relative_pos"),
                ]
            )

            out_cols = [pl.col(c) for c in seq_cols]
            out_cols.append(pl.col("sequence_id"))
            if expand_skills:
                out_cols.append(pl.col("skill"))
            out_cols.append(pl.col("relative_pos").alias("seq_pos"))
            parts.append(b.select(out_cols).sort("sequence_id", "seq_pos"))

            next_new_id += n_new
            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    f"  processed {batch_idx + 1} batches, "
                    f"{next_new_id} sub-sequences so far"
                )

        if not parts:
            empty = self.sequence_data.head(0).select(seq_cols)
            empty = empty.with_columns(
                pl.lit(None, dtype=pl.Int32).alias("sequence_id")
            )
            if expand_skills:
                empty = empty.with_columns(pl.lit(None, dtype=pl.Int32).alias("skill"))
            return empty.with_columns(pl.lit(None, dtype=pl.Int64).alias("seq_pos"))

        return pl.concat(parts, how="vertical")

    def build_split_question_sequence_data(self) -> None:
        """Build split sequence data by question.

        Splits user sequences longer than ``max_seq_len`` into multiple
        sub-sequences, drops splits shorter than ``min_seq_len``, and assigns
        a new dense ``sequence_id`` to each retained split.

        Processing is done in user-aligned batches to bound peak memory,
        while producing output identical to whole-frame processing.
        """
        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        self.split_question_sequence_data = self._build_split_sequences(
            expand_skills=False
        )
        final_num_users = self.split_question_sequence_data["sequence_id"].n_unique()
        logger.debug(f"Split into {final_num_users} question sub-sequences")

    def build_split_skill_sequence_data(self) -> None:
        """Build split skill sequence data.

        Expands question sequences into skill sequences (one question may map
        to multiple skills), preserving the question column, then splits long
        sequences by ``max_seq_len`` and drops splits shorter than
        ``min_seq_len``.

        Processing is done in user-aligned batches to bound peak memory,
        while producing output identical to whole-frame processing.

        Output columns: original sequence_data columns + sequence_id + skill + seq_pos
        """
        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )
        if not self.relation_data or "question_skill" not in self.relation_data:
            raise ValueError("question_skill relation not available.")

        self.split_skill_sequence_data = self._build_split_sequences(expand_skills=True)
        logger.debug(
            f"Split into {self.split_skill_sequence_data['sequence_id'].n_unique()} "
            f"skill sub-sequences"
        )

    def build_windowlate_data(self) -> None:
        """Build windowlate evaluation samples for windowlate_auc_mean scoring.

        Data is streamed directly to file in this method.
        """
        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )
        if not self.relation_data or "question_skill" not in self.relation_data:
            raise ValueError("question_skill relation not available.")
        if "fold" not in self.sequence_data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call add_kfold_labels() first."
            )

        # Filter test set data
        test_data = self.sequence_data.filter(pl.col("fold") == -1)
        if len(test_data) == 0:
            raise ValueError("No test-set interactions (fold == -1) found")

        max_seq_len = self.args.max_seq_len
        logger.info(f"Building windowlate data (max_seq_len={max_seq_len})...")

        # Prepare output path
        os.makedirs(self.data_folder, exist_ok=True)
        output_path = os.path.join(
            self.data_folder, f"{self.dataset}_windowlate.parquet"
        )

        # Get configuration parameters
        users_per_batch = getattr(self.args, "windowlate_users_per_batch", 1)

        # Build and save directly to file
        WindowlateProcessor.build(
            test_data=test_data,
            question_data=self.relation_data["question_skill"],
            max_seq_len=max_seq_len,
            output_path=output_path,
            users_per_batch=users_per_batch,
        )

    def add_kfold_labels(self, n_splits: int = 5, test_ratio: float = 0.2) -> None:
        """Add K-fold cross-validation labels with test set separation.

        Ensures all data from the same user stays in the same fold
        to prevent data leakage.

        The process:
        1. Split users into test set and non-test set
        2. Test set users are labeled with -1
        3. Non-test set users are split into n_splits folds (0 to n_splits-1)

        Args:
            n_splits: Number of folds for cross-validation (default: 5).
            test_ratio: Ratio of users to allocate to test set (default: 0.2).

        Returns:
            DataFrame with added 'fold' column (values: -1 for test set, 0 to n_splits-1 for train/val).

        Raises:
            ValueError: If sequence_data is not loaded.
        """
        if test_ratio > 1 or test_ratio < 0:
            raise ValueError("Test ratio should within 0~1.")

        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        # Get unique user IDs
        unique_users = self.sequence_data["user"].unique().sort()
        num_users = len(unique_users)
        num_test_users = int(num_users * test_ratio)

        # Shuffle user IDs (seeded local generator, like the KFold below, so
        # the test/cv split is reproducible per data source and does not touch
        # the global numpy RNG state).
        rng = np.random.default_rng(self.seed)
        user_indices = np.arange(num_users)
        rng.shuffle(user_indices)
        # Get indices of non-test users after shuffling
        non_test_indices = user_indices[num_test_users:]
        # Initialize fold assignment array
        fold_assignment = np.full(num_users, -1, dtype=np.int32)
        # Perform K-fold cross-validation on non-test users
        logger.debug(
            f"Splitting {num_users - num_test_users} users into {n_splits} folds..."
        )
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for fold_idx, (_, val_indices) in enumerate(kfold.split(non_test_indices)):
            fold_assignment[non_test_indices[val_indices]] = fold_idx

        user_fold_map = pl.DataFrame(
            {"user": unique_users, "fold": pl.Series(fold_assignment, dtype=pl.Int32)}
        )

        self.sequence_data = self.sequence_data.join(
            user_fold_map, on="user", how="left"
        )
        self.update_metadata("kfold_n_splits", n_splits)
        self.update_metadata("test_ratio", test_ratio)

        logger.info(
            f"Added K-fold labels with n_splits={n_splits}, test_ratio={test_ratio}"
        )

    def get_user_stats(self) -> pl.DataFrame:
        """Compute user statistics: attempts, correct count, skill count, correct rate.

        Returns:
            DataFrame with columns: user, attempts, correct, skill_count, correct_rate.

        Raises:
            ValueError: If sequence_data is not loaded.
        """
        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        user_stats = self.sequence_data.group_by("user").agg(
            pl.len().alias("attempts"),
            pl.col("label").sum().alias("correct"),
        )

        user_skill = self.sequence_data.select(["user", "question"]).join(
            self.relation_data["question_skill"].select(["question", "skill"]),
            on="question",
            how="left",
        )
        skill_stats = user_skill.group_by("user").agg(
            pl.col("skill").n_unique().alias("skill_count")
        )

        user_stats = user_stats.join(skill_stats, on="user", how="left").with_columns(
            [
                pl.col("skill_count").fill_null(0).cast(pl.Int32),
                (pl.col("correct") / pl.col("attempts")).alias("correct_rate"),
            ]
        )

        return user_stats

    def sample(
        self,
        sample_size: int | None = None,
        sample_ratio: float | None = None,
        sample_strategy: str = "random",
        attempts_bins: Sequence[float] = [20, 100],
        correct_bins: list[float] = [0.4, 0.8],
    ) -> None:
        """Sample dataset by users or interactions.

        Args:
            sample_size: Absolute sample count. For random/stratified: number
                of users. For time: number of interactions.
            sample_ratio: Sample ratio (0.0-1.0). Overrides sample_size if set.
            sample_strategy: Sampling strategy (random, stratified, time).
            attempts_bins: Attempt count bin edges for stratified sampling.
            correct_bins: Correct rate bin edges for stratified sampling.

        Raises:
            ValueError: If neither sample_size nor sample_ratio is provided,
                or if the computed size exceeds available data.
        """
        user_stats = self.get_user_stats()
        total_users = len(user_stats)
        original_records = len(self.sequence_data)

        # Compute n_samples from sample_ratio or sample_size
        if sample_ratio is not None and sample_size is not None:
            raise ValueError(
                "Cannot specify both sample_ratio and sample_size, use one only"
            )
        elif sample_ratio is not None:
            if not 0.0 < sample_ratio <= 1.0:
                raise ValueError(
                    f"sample_ratio must be in (0.0, 1.0], got {sample_ratio}"
                )
            total = original_records if sample_strategy == "time" else total_users
            n_samples = max(1, int(total * sample_ratio))
            logger.info(
                f"Sampling {sample_ratio:.2%} of {total} "
                f"({'interactions' if sample_strategy == 'time' else 'users'}) "
                f"= {n_samples}, strategy={sample_strategy}"
            )
        elif sample_size is not None:
            n_samples = sample_size
            logger.info(
                f"Sampling {n_samples} "
                f"({'interactions' if sample_strategy == 'time' else 'users'}) "
                f"from dataset, strategy={sample_strategy}"
            )
        else:
            return

        if sample_strategy in ("random", "stratified"):
            if n_samples > total_users:
                raise ValueError(
                    f"Requested sample size ({n_samples}) exceeds total users ({total_users})"
                )
            if n_samples == total_users:
                logger.info("Sample size equals total users, skipping sampling.")
                return

        if sample_strategy == "random":
            sampled_users = (
                user_stats.sort("user")
                .sample(n=n_samples, seed=self.seed)
                .select("user")
                .to_series()
                .to_list()
            )
            self._apply_sampling_to_data(
                sampled_users,
                n_samples,
                total_users,
                original_records,
                stratify=False,
            )
        elif sample_strategy == "stratified":
            sampled_users = self._sample_users_stratified(
                user_stats, n_samples, attempts_bins, correct_bins
            )
            self._apply_sampling_to_data(
                sampled_users,
                n_samples,
                total_users,
                original_records,
                stratify=True,
            )
        elif sample_strategy == "time":
            self._apply_time_sampling(n_samples, total_users, original_records)
        else:
            raise ValueError(f"Unsupported sample strategy: {sample_strategy}")

    def _sample_users_stratified(
        self,
        user_stats: pl.DataFrame,
        n_samples: int,
        attempts_bins: Sequence[float],
        correct_bins: list[float],
    ) -> list[int]:
        """Perform stratified sampling on users."""
        user_stats = user_stats.with_columns(
            [
                (
                    self._make_bin_expr("attempts", attempts_bins) * 3
                    + self._make_bin_expr("correct_rate", correct_bins)
                ).alias("strata"),
                ((pl.col("user").cast(pl.Int64) + self.seed) % 2**31)
                .cast(pl.UInt32)
                .alias("rand_key"),
            ]
        )

        strata_info = user_stats.group_by("strata").agg(pl.len().alias("count"))

        total_in_strata = strata_info.select(pl.sum("count")).item()
        strata_info = strata_info.with_columns(
            ((pl.col("count") / total_in_strata * n_samples).cast(pl.Int32))
            .clip(1)
            .alias("quota")
        )

        strata_info = strata_info.sort("strata")
        logger.info(f"Created {len(strata_info)} strata for stratified sampling.")

        sampled_users = (
            user_stats.join(strata_info.select(["strata", "quota"]), on="strata")
            .sort(["strata", "rand_key"])
            .with_columns(
                pl.int_range(pl.len(), dtype=pl.UInt32).over("strata").alias("rank")
            )
            .filter(pl.col("rank") < pl.col("quota"))
            .select("user")
        )

        n_sampled = len(sampled_users)
        if n_sampled != n_samples:
            if n_sampled > n_samples:
                sampled_users = sampled_users.sample(n=n_samples, seed=self.seed)
            else:
                unsampled = user_stats.join(
                    sampled_users, on="user", how="anti"
                ).select("user")
                additional = unsampled.sample(n=n_samples - n_sampled, seed=self.seed)
                sampled_users = pl.concat([sampled_users, additional])

        return sampled_users.to_series().to_list()

    def _make_bin_expr(self, col: str, bins: Sequence[float]) -> pl.Expr:
        """Create binning expression for a column."""
        return (
            pl.when(pl.col(col) <= bins[0])
            .then(pl.lit(0, dtype=pl.Int8))
            .when(pl.col(col) <= bins[1])
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(2, dtype=pl.Int8))
        )

    def _apply_sampling_to_data(
        self,
        sampled_users: list[int],
        n_samples: int,
        total_users: int,
        original_records: int,
        stratify: bool = True,
    ) -> None:
        """Apply sampled users to sequence and question data with ID remapping."""
        user_stats = self.get_user_stats()
        strata_distribution = self._compute_strata_distribution(
            sampled_users, user_stats
        )

        self.sequence_data = self.sequence_data.filter(
            pl.col("user").is_in(sampled_users)
        )

        self._remap_user_ids()
        self._remap_question_ids()

        num_users = self.sequence_data.select(pl.col("user").n_unique()).item()
        qs = self.relation_data["question_skill"]
        num_questions = qs.select(pl.col("question").n_unique()).item()
        num_skills = qs.select(pl.col("skill").n_unique()).item()
        sampled_records = len(self.sequence_data)

        sampling_config = {
            "n_samples_requested": n_samples,
            "n_samples_actual": len(sampled_users),
            "sampling_ratio": len(sampled_users) / total_users,
            "stratify": stratify,
            "attempts_bins": [20, 100],
            "correct_bins": [0.4, 0.8],
        }

        sampling_stats = {
            "original_users": total_users,
            "sampled_users": len(sampled_users),
            "original_records": original_records,
            "sampled_records": sampled_records,
            "strata_distribution": strata_distribution,
        }

        self.update_metadata("sampled", True)
        self.update_metadata("sampling_config", sampling_config)
        self.update_metadata("sampling_stats", sampling_stats)
        self.update_metadata("num_users", int(num_users))
        self.update_metadata("num_questions", int(num_questions))
        self.update_metadata("num_skills", int(num_skills))

        logger.info(
            f"Sampling complete: {len(sampled_users)}/{total_users} users, "
            f"{sampled_records}/{original_records} records"
        )
        logger.info(f"Sampling ratio: {sampling_config['sampling_ratio']:.2%}")

    def _apply_time_sampling(
        self,
        n_samples: int,
        total_users: int,
        original_records: int,
    ) -> None:
        """Sample by taking the earliest N interactions sorted by timestamp."""
        if n_samples > original_records:
            raise ValueError(
                f"Requested sample size ({n_samples}) exceeds total records ({original_records})"
            )

        self.sequence_data = self.sequence_data.sort(
            self._sort_columns(["timestamp", "user"])
        ).head(n_samples)

        self._remap_user_ids()
        self._remap_question_ids()

        num_users = self.sequence_data.select(pl.col("user").n_unique()).item()
        qs = self.relation_data["question_skill"]
        num_questions = qs.select(pl.col("question").n_unique()).item()
        num_skills = qs.select(pl.col("skill").n_unique()).item()
        sampled_records = len(self.sequence_data)

        sampling_config = {
            "n_samples_requested": n_samples,
            "n_samples_actual": sampled_records,
            "sampling_ratio": sampled_records / original_records,
            "strategy": "time",
        }

        sampling_stats = {
            "original_users": total_users,
            "sampled_users": num_users,
            "original_records": original_records,
            "sampled_records": sampled_records,
        }

        self.update_metadata("sampled", True)
        self.update_metadata("sampling_config", sampling_config)
        self.update_metadata("sampling_stats", sampling_stats)
        self.update_metadata("num_users", int(num_users))
        self.update_metadata("num_questions", int(num_questions))
        self.update_metadata("num_skills", int(num_skills))

        logger.info(
            f"Time sampling complete: {sampled_records}/{original_records} records, "
            f"{num_users}/{total_users} users"
        )
        logger.info(f"Sampling ratio: {sampling_config['sampling_ratio']:.2%}")

    def _remap_user_ids(self) -> None:
        """Remap user IDs to consecutive integers starting from 0."""
        user_id_map = (
            self.sequence_data.select(pl.col("user").unique())
            .sort("user")
            .with_row_index("new_user_id")
            .select([pl.col("user"), pl.col("new_user_id").cast(pl.Int32)])
        )
        self.sequence_data = (
            self.sequence_data.join(user_id_map, on="user", how="left")
            .drop("user")
            .rename({"new_user_id": "user"})
        )

    def _remap_question_ids(self) -> None:
        """Filter and remap question IDs and entity IDs to consecutive integers."""
        active_questions = self.sequence_data.select(pl.col("question").unique())

        # Step 1: Filter all relations to only active questions
        for name in list(self.relation_data.keys()):
            df = self.relation_data[name]
            if "question" in df.columns:
                self.relation_data[name] = df.join(
                    active_questions, on="question", how="semi"
                )

        # Step 2: Remap question IDs (consistent across all relations)
        question_id_map = (
            active_questions.sort("question")
            .with_row_index("new_question_id")
            .select([pl.col("question"), pl.col("new_question_id").cast(pl.Int32)])
        )
        for name in list(self.relation_data.keys()):
            df = self.relation_data[name]
            if "question" in df.columns:
                self.relation_data[name] = (
                    df.join(question_id_map, on="question", how="left")
                    .drop("question")
                    .rename({"new_question_id": "question"})
                )
        self.sequence_data = (
            self.sequence_data.join(question_id_map, on="question", how="left")
            .drop("question")
            .rename({"new_question_id": "question"})
        )

        # Step 3: Remap entity IDs in each relation
        for name in list(self.relation_data.keys()):
            df = self.relation_data[name]
            entity_col = next(c for c in df.columns if c != "question")
            entity_id_map = (
                df.select(pl.col(entity_col).unique())
                .sort(entity_col)
                .with_row_index(f"new_{entity_col}")
                .select(
                    [
                        pl.col(entity_col),
                        pl.col(f"new_{entity_col}").cast(pl.Int32),
                    ]
                )
            )
            self.relation_data[name] = (
                df.join(entity_id_map, on=entity_col, how="left")
                .drop(entity_col)
                .rename({f"new_{entity_col}": entity_col})
            )

    def _compute_strata_distribution(
        self, sampled_users: list[int], user_stats: pl.DataFrame
    ) -> dict[str, dict[str, int]]:
        """Compute distribution of users across strata."""

        def _strata_to_str(s: int) -> str:
            return f"{s // 3}_{s % 3}"

        strata_to_str = _strata_to_str

        strata_cols = user_stats.columns
        if "strata" not in strata_cols:
            return {}

        strata_info = user_stats.group_by("strata").agg(pl.len().alias("count"))
        strata_distribution = {
            strata_to_str(row["strata"]): {"original": row["count"], "sampled": 0}
            for row in strata_info.iter_rows(named=True)
        }

        sampled_strata = (
            user_stats.filter(pl.col("user").is_in(sampled_users))
            .group_by("strata")
            .agg(pl.len().alias("sampled"))
        )

        for row in sampled_strata.iter_rows(named=True):
            strata_str = strata_to_str(row["strata"])
            if strata_str in strata_distribution:
                strata_distribution[strata_str]["sampled"] = row["sampled"]

        return strata_distribution


@overload
def exclude_short_sequences(data: pl.LazyFrame, min_seq_len: int) -> pl.LazyFrame: ...


@overload
def exclude_short_sequences(data: pl.DataFrame, min_seq_len: int) -> pl.DataFrame: ...


def exclude_short_sequences(
    data: pl.DataFrame | pl.LazyFrame, min_seq_len: int
) -> pl.DataFrame | pl.LazyFrame:
    """Filter out users with sequence length less than min_seq_len.

    Args:
        data: Polars DataFrame or LazyFrame.
        min_seq_len: Minimum sequence length.

    Returns:
        DataFrame or LazyFrame of same type as input.
    """
    if min_seq_len > 1:
        user_frame = (
            data.group_by("user")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= min_seq_len)
            .select("user")
        )
        valid_users = (
            user_frame.collect().to_series()
            if isinstance(user_frame, pl.LazyFrame)
            else user_frame.to_series()
        )
        data = data.filter(pl.col("user").is_in(valid_users))

    return data


__all__ = [
    "DataSource",
    "exclude_short_sequences",
]
