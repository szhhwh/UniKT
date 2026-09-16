"""Tests for dataset on-disk state inspection: empty/downloaded/ready matrix."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from utils.dataset_status import (
    dataset_status,
    has_raw,
    is_processed,
    list_dataset_statuses,
)


def _make_raw(
    base: Path, dataset: str = "ds", files: Sequence[str] = ("a.csv",)
) -> Path:
    raw = base / dataset / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in files:
        (raw / f).write_text("x", encoding="utf-8")
    return raw


def _make_processed(base: Path, dataset: str = "ds") -> Path:
    folder = base / dataset
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(
        json.dumps({"sequence_data_md5": "abc"}), encoding="utf-8"
    )
    return folder


@pytest.fixture(autouse=True)
def _fixed_supported_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the test hermetic: no lazy discovery scan of utils/data_process.
    monkeypatch.setattr(
        "utils.dataset_status.get_supported_datasets",
        lambda: ["ds", "other"],
    )


class TestStatusMatrix:
    def test_missing_folder_is_empty(self, tmp_path: Path) -> None:
        assert dataset_status("ds", tmp_path) == "empty"

    def test_empty_raw_dir_is_empty(self, tmp_path: Path) -> None:
        (tmp_path / "ds" / "raw").mkdir(parents=True)
        assert dataset_status("ds", tmp_path) == "empty"

    def test_nonempty_raw_is_downloaded(self, tmp_path: Path) -> None:
        _make_raw(tmp_path)
        assert dataset_status("ds", tmp_path) == "downloaded"

    def test_processed_marker_is_ready(self, tmp_path: Path) -> None:
        _make_raw(tmp_path)
        _make_processed(tmp_path)
        assert dataset_status("ds", tmp_path) == "ready"

    def test_ready_short_circuits_without_raw(self, tmp_path: Path) -> None:
        # Processed datasets may drop raw files; they stay usable.
        _make_processed(tmp_path)
        assert not has_raw(tmp_path, "ds")
        assert dataset_status("ds", tmp_path) == "ready"

    def test_download_marker_without_processed_is_downloaded(
        self, tmp_path: Path
    ) -> None:
        _make_raw(tmp_path)
        (tmp_path / "ds" / "metadata.json").write_text(
            json.dumps({"url": "http://x"}), encoding="utf-8"
        )
        assert dataset_status("ds", tmp_path) == "downloaded"


class TestRobustness:
    def test_corrupt_metadata_json_not_processed(self, tmp_path: Path) -> None:
        _make_raw(tmp_path)
        (tmp_path / "ds" / "metadata.json").write_text("{not json", encoding="utf-8")
        assert is_processed(tmp_path, "ds") is False
        assert dataset_status("ds", tmp_path) == "downloaded"

    def test_unreadable_metadata_returns_false(self, tmp_path: Path) -> None:
        # metadata.json as a directory: exists() is True but read raises
        # IsADirectoryError (an OSError) -> swallowed to False.
        _make_raw(tmp_path)
        (tmp_path / "ds" / "metadata.json").mkdir()
        assert is_processed(tmp_path, "ds") is False
        assert dataset_status("ds", tmp_path) == "downloaded"

    def test_dataset_name_lowercased(self, tmp_path: Path) -> None:
        _make_raw(tmp_path, dataset="mixedcase")
        assert dataset_status("MixedCase", tmp_path) == "downloaded"
        assert dataset_status("MIXEDCASE", tmp_path) == "downloaded"


class TestListDatasetStatuses:
    def test_one_row_per_supported_dataset(self, tmp_path: Path) -> None:
        _make_raw(tmp_path, dataset="ds")
        rows = list_dataset_statuses(tmp_path)
        assert rows == [
            {"name": "ds", "status": "downloaded"},
            {"name": "other", "status": "empty"},
        ]
