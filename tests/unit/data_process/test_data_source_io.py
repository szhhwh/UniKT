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
from utils.data_process.windowlate_processor import WindowlateProcessor

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

    @pytest.mark.parametrize(
        "data_type",
        ["sequence", "split_question_sequence", "split_skill_sequence", "windowlate"],
    )
    def test_load_rejects_changed_processed_artifact(
        self,
        make_data_source: Callable[..., DataSource],
        tmp_path: Path,
        data_type: str,
    ) -> None:
        target = tmp_path / f"stub_{data_type}.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(target)
        source = make_data_source(data_folder=str(tmp_path))
        source.metadata[f"{data_type}_data_md5"] = source.compute_md5(str(target))
        pl.DataFrame({"a": [2]}).write_parquet(target)

        with pytest.raises(ValueError, match="MD5 mismatch"):
            source._load_data(data_type)

    def test_saving_replaces_cached_frames_and_split_schemas(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        source = make_data_source(
            sequence_data=make_sequence_frame([0, 0], [0, 0], labels=[0, 1])
        )
        source.set_sequence_data(source.sequence_data)
        source.build_split_question_sequence_data()
        source.build_split_skill_sequence_data()
        source.save_data()
        before = source.get_split_question_sequence_data()
        source.get_sequence_data()

        source.set_sequence_data(
            source.sequence_data.with_columns(
                (1 - pl.col("label")).alias("label"), pl.lit(2).alias("hint_count")
            )
        )
        source.build_split_question_sequence_data()
        source.build_split_skill_sequence_data()
        source.save_data()

        after = source.get_split_question_sequence_data(required=("hint_count",))
        assert after["label"].to_list() == [1, 0]
        assert before["label"].to_list() == [0, 1]
        assert after["hint_count"].to_list() == [2, 2]
        assert source.get_sequence_data()["label"].to_list() == [1, 0]


class TestWindowlateLifecycle:
    @pytest.fixture(autouse=True)
    def restore_window_columns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(WindowlateProcessor, "EXTRA_COLUMNS", [])
        monkeypatch.setattr(WindowlateProcessor, "EXTRA_DTYPES", {})

    def test_new_events_clear_processing_settings_and_preserve_download_metadata(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
    ) -> None:
        source = make_data_source(
            metadata={
                "raw_data_md5": "download",
                "kfold_n_splits": 5,
                "test_ratio": 0.2,
                "sampled": True,
                "sampling_config": {"ratio": 0.1},
                "sampling_stats": {"rows": 10},
            },
        )
        source.set_sequence_data(
            make_sequence_frame([0], [0]).with_columns(
                pl.Series("skills", [[0]], dtype=pl.List(pl.Int32))
            )
        )
        assert source.metadata == {"raw_data_md5": "download"}

    def test_new_events_replace_previously_cached_question_skill_relation(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        tmp_path: Path,
    ) -> None:
        pl.DataFrame({"question": [0], "skill": [0]}).write_parquet(
            tmp_path / "stub_relation_question_skill.parquet"
        )
        source = make_data_source()
        assert source.get_relation("question_skill")["skill"].to_list() == [0]
        source.set_sequence_data(
            make_sequence_frame([0], [0]).with_columns(
                pl.Series("skills", [[1]], dtype=pl.List(pl.Int32))
            )
        )
        assert source.get_relation("question_skill")["skill"].to_list() == [1]

    @pytest.mark.parametrize("change", ["events", "folds", "sampling"])
    def test_changed_inputs_revoke_cached_windows_without_reloading_old_metadata(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        tmp_path: Path,
        change: str,
    ) -> None:
        source = make_data_source(
            sequence_data=make_sequence_frame(list(range(8)), [0] * 8),
            metadata={
                "windowlate_data_md5": "old",
                "max_windowlate_seq_len": 10,
            },
        )
        source.save_metadata()
        source._data_cache[("windowlate", None)] = pl.DataFrame({"skill": [99]}).lazy()
        if change == "events":
            source.set_sequence_data(source.sequence_data)
        elif change == "folds":
            source.add_kfold_labels(n_splits=2)
        else:
            source._remap_question_ids()

        assert "windowlate_data_md5" not in source.metadata
        assert "max_windowlate_seq_len" not in source.metadata
        assert ("windowlate", None) not in source._data_cache
        with pytest.raises(ValueError, match="needs windowlate preprocessing"):
            source.get_windowlate_data()
        assert (
            json.loads((tmp_path / "metadata.json").read_text())["windowlate_data_md5"]
            == "old"
        )

    def test_build_registers_windows_before_save_and_persists_registration(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        tmp_path: Path,
    ) -> None:
        source = make_data_source(
            sequence_data=make_sequence_frame([0, 0, 0], [0, 0, 0]).with_columns(
                pl.lit(-1).alias("fold"),
                pl.Series("skills", [[0], [1], [0, 1]], dtype=pl.List(pl.Int32)),
            )
        )
        source.set_sequence_data(source.sequence_data)
        source.build_split_question_sequence_data()
        source.build_split_skill_sequence_data()
        source.build_windowlate_data()

        windows = source.get_windowlate_data().collect()
        assert windows.filter(pl.col("mask") == 1)["skill"].to_list() == [1, 0, 1]
        checksum = source.compute_md5(str(tmp_path / "stub_windowlate.parquet"))
        assert source.metadata["windowlate_data_md5"] == checksum

        source.save_data()
        reader = make_data_source()
        reader.load_metadata()
        assert reader.metadata["windowlate_data_md5"] == checksum
        assert reader.get_windowlate_data().collect().equals(windows)

    def test_failed_build_revokes_previous_registration(
        self,
        make_data_source: Callable[..., DataSource],
        make_sequence_frame: Callable[..., pl.DataFrame],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from utils.data_process import data_source

        class FailingWindows:
            @staticmethod
            def build(**kwargs: object) -> int:
                raise RuntimeError("window generation failed")

        source = make_data_source(
            sequence_data=make_sequence_frame([0, 0], [0, 0]).with_columns(
                pl.lit(-1).alias("fold")
            ),
            metadata={"windowlate_data_md5": "old", "max_windowlate_seq_len": 10},
        )
        monkeypatch.setattr(data_source, "WindowlateProcessor", FailingWindows)
        with pytest.raises(RuntimeError, match="window generation failed"):
            source.build_windowlate_data()
        with pytest.raises(ValueError, match="needs windowlate preprocessing"):
            source.get_windowlate_data()


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

    def test_metadata_load_is_explicit(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        (tmp_path / "metadata.json").write_text(json.dumps({"from_disk": True}))
        ds = make_data_source(data_folder=str(tmp_path), metadata={})
        assert ds.get_metadata() == {}
        ds.load_metadata()
        assert ds.get_metadata("from_disk") is True

    def test_new_source_loads_saved_metadata(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        source_class = type(make_data_source())
        folder = tmp_path / "stub"
        folder.mkdir()
        (folder / "metadata.json").write_text(json.dumps({"num_users": 3}))
        source = source_class("stub", str(tmp_path))
        assert source.get_metadata("num_users") == 3

    def test_reloading_metadata_refreshes_cached_artifacts(
        self, make_data_source: Callable[..., DataSource], tmp_path: Path
    ) -> None:
        artifact = tmp_path / "stub_sequence.parquet"
        pl.DataFrame({"label": [0]}).write_parquet(artifact)
        source = make_data_source()
        assert source.get_sequence_data()["label"].to_list() == [0]
        source._split_schema_cache["split_question_sequence"] = frozenset({"label"})
        pl.DataFrame({"label": [1]}).write_parquet(artifact)
        (tmp_path / "metadata.json").write_text(
            json.dumps({"sequence_data_md5": source.compute_md5(str(artifact))})
        )
        source.load_metadata()
        assert source._split_schema_cache == {}
        assert source.get_sequence_data()["label"].to_list() == [1]


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
