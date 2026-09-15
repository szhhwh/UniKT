"""Tests for DataSource archive extraction, _load_data caching, metadata, md5."""

import gzip
import hashlib
import io
import json
import os
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from utils.data_process.data_source import DataSource

# --- _extract_archive -------------------------------------------------------------


def _add_tar_text(tf: tarfile.TarFile, name: str, content: str) -> None:
    info = tarfile.TarInfo(name)
    data = content.encode()
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


class TestExtractArchive:
    def test_zip_archive(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("a.txt", "hello")
            zf.writestr("sub/b.txt", "world")
        target = tmp_path / "raw_zip"
        target.mkdir()

        make_data_source()._extract_archive(str(archive), "bundle.zip", str(target))
        assert (target / "a.txt").read_text() == "hello"
        assert (target / "sub" / "b.txt").read_text() == "world"

    @pytest.mark.parametrize("file_name", ["bundle.tar.gz", "bundle.tgz"])
    def test_tar_gz_archive(
        self,
        make_data_source: Callable[..., DataSource],
        tmp_path: Path,
        file_name: str,
    ) -> None:
        archive = tmp_path / file_name
        with tarfile.open(archive, "w:gz") as tf:
            _add_tar_text(tf, "x.txt", "tar-gz-content")
        target = tmp_path / "raw_tgz"
        target.mkdir()

        make_data_source()._extract_archive(str(archive), file_name, str(target))
        assert (target / "x.txt").read_text() == "tar-gz-content"

    def test_plain_tar_archive(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        archive = tmp_path / "bundle.tar"
        with tarfile.open(archive, "w") as tf:
            _add_tar_text(tf, "y.txt", "tar-content")
        target = tmp_path / "raw_tar"
        target.mkdir()

        make_data_source()._extract_archive(str(archive), "bundle.tar", str(target))
        assert (target / "y.txt").read_text() == "tar-content"

    def test_single_gz_file(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        archive = tmp_path / "notes.txt.gz"
        archive.write_bytes(gzip.compress(b"gz-content"))
        target = tmp_path / "raw_gz"
        target.mkdir()

        make_data_source()._extract_archive(str(archive), "notes.txt.gz", str(target))
        # Decompressed beside the stripped extension, named after the archive.
        assert (target / "notes.txt").read_text() == "gz-content"

    def test_plain_file_copied(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        archive = tmp_path / "raw.csv"
        archive.write_text("a,b\n1,2\n")
        target = tmp_path / "raw_plain"
        target.mkdir()

        make_data_source()._extract_archive(str(archive), "raw.csv", str(target))
        assert (target / "raw.csv").read_text() == "a,b\n1,2\n"
        # The original archive is untouched (copy, not move).
        assert archive.read_text() == "a,b\n1,2\n"


# --- _load_data caching -----------------------------------------------------------


class TestLoadData:
    def test_second_call_uses_cache_without_disk(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(data_dir / "stub_sequence.parquet")

        ds = make_data_source(data_folder=str(data_dir))
        # "sequence" is configured eager; the union narrow is via cast (no runtime assert).
        first = cast(pl.DataFrame, ds._load_data("sequence"))
        os.remove(data_dir / "stub_sequence.parquet")

        # File gone: a disk re-read would raise, the cache must not re-read.
        second = cast(pl.DataFrame, ds._load_data("sequence"))
        assert second is first
        assert first["a"].to_list() == [1, 2, 3]

    def test_lazy_config_returns_lazyframe(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pl.DataFrame({"a": [1]}).write_parquet(data_dir / "stub_windowlate.parquet")

        ds = make_data_source(data_folder=str(data_dir))
        assert isinstance(ds._load_data("windowlate"), pl.LazyFrame)

    def test_unknown_data_type_raises(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        with pytest.raises(ValueError, match="Unknown data type: bogus"):
            make_data_source()._load_data("bogus")


# --- metadata JSON round-trip -------------------------------------------------------


class TestMetadata:
    def test_round_trip_via_disk(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        ds = make_data_source(data_folder=str(tmp_path))
        ds.update_metadata("num_users", 3)
        ds.update_metadata("ratio", 0.25)

        ds.save_metadata()
        assert (tmp_path / "metadata.json").exists()
        on_disk = json.loads((tmp_path / "metadata.json").read_text())
        # save_metadata stamps the dataset identity keys itself.
        assert on_disk == {
            "num_users": 3,
            "ratio": 0.25,
            "dataset": "stub",
            "data_base_path": str(tmp_path),
        }

        # A fresh instance re-reads the same entries from disk.
        other = make_data_source(data_folder=str(tmp_path))
        other.load_metadata()
        assert other.get_metadata() == ds.get_metadata()
        assert other.get_metadata("num_users") == 3

    def test_get_metadata_missing_key_raises(
        self, make_data_source: Callable[..., DataSource]
    ) -> None:
        ds = make_data_source(metadata={"present": 1})
        with pytest.raises(
            KeyError, match="Metadata key 'absent' not found in dataset 'stub'"
        ):
            ds.get_metadata("absent")

    def test_load_metadata_missing_file_raises(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        ds = make_data_source(data_folder=str(tmp_path / "nowhere"))
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            ds.load_metadata()

    def test_empty_metadata_triggers_disk_load(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        # get_metadata on an empty in-memory dict falls back to the file.
        (tmp_path / "metadata.json").write_text(json.dumps({"from_disk": True}))
        ds = make_data_source(data_folder=str(tmp_path), metadata={})
        assert ds.get_metadata("from_disk") is True


# --- compute_md5 --------------------------------------------------------------------


class TestComputeMd5:
    def test_multichunk_file_matches_hashlib(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        # 76800 bytes > the 65536 read chunk: exercises the chunked loop.
        content = bytes(range(256)) * 300
        target = tmp_path / "blob.bin"
        target.write_bytes(content)

        md5 = make_data_source().compute_md5(str(target))
        assert md5 == hashlib.md5(content).hexdigest()
        # Sanity: the payload really spans more than one chunk.
        assert len(content) > 65536

    def test_empty_file(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        target = tmp_path / "empty.bin"
        target.write_bytes(b"")
        assert (
            make_data_source().compute_md5(str(target)) == hashlib.md5(b"").hexdigest()
        )
