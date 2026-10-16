"""Shared fixtures for model_data tests.

Two groups of doubles:

- ``StubDataSource`` / ``write_windowlate_parquet`` / ``make_skill_model_data``
  back the windowlate dataloader tests.
- ``EventSource`` and friends model a split frame whose ``skills`` column holds
  per-interaction annotation lists (event boundaries), which is what the
  packed-sequence and split-loading tests assert against.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.skill_model_data import SkillModelData

if TYPE_CHECKING:
    from collections.abc import Callable

    from utils.config.run_config import RunConfig

_CANONICAL_WINDOWLATE_COLUMNS = (
    ("sample_id", pa.int32()),
    ("position", pa.int32()),
    ("skill", pa.int32()),
    ("question", pa.int32()),
    ("response", pa.int8()),
    ("mask", pa.int8()),
    ("user_id", pa.int32()),
    ("group_id", pa.int64()),
    ("true_label", pa.int8()),
)


class StubDataSource:
    """Minimal DataSource stand-in: metadata dict + in-memory data frames.

    Bypasses the md5/integrity plumbing so tests exercise only the id-space
    propagation logic under test.
    """

    def __init__(
        self,
        split_skill_sequence_data: pl.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = "stub"
        self.data_folder = "/nonexistent"
        self._split_skill_sequence_data = split_skill_sequence_data
        self._metadata = metadata or {}

    def get_split_skill_sequence_data(
        self,
        required: tuple[str, ...] = (),
        optional: tuple[str, ...] = (),
    ) -> pl.DataFrame | None:
        return self._split_skill_sequence_data

    def get_metadata(self, key: str | None = None) -> Any:
        if key is not None and key not in self._metadata:
            raise KeyError(f"Metadata key '{key}' not found")
        return self._metadata if key is None else self._metadata[key]


@pytest.fixture
def write_windowlate_parquet(tmp_path: Path) -> Callable[..., str]:
    """Return a writer bound to tmp_path; writes the canonical column schema."""

    def _write(filename: str, samples: dict[str, list]) -> str:
        schema = pa.schema(list(_CANONICAL_WINDOWLATE_COLUMNS))
        table = pa.table(
            {
                name: pa.array(samples[name], type=schema.field(name).type)
                for name in schema.names
            },
            schema=schema,
        )
        path = str(tmp_path / filename)
        pq.write_table(table, path)
        return path

    return _write


@pytest.fixture
def make_skill_model_data() -> Callable[..., SkillModelData]:
    """Factory: bind a concrete SkillModelData to a StubDataSource."""

    class _ConcreteSkillModelData(SkillModelData):
        def prepare_data(self, rc: RunConfig) -> None:
            raise NotImplementedError("stub; call build/load methods directly")

    def _make(
        split_frame: pl.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SkillModelData:
        return _ConcreteSkillModelData(
            # Stub implements the duck-typed surface SkillModelData reads;
            # it deliberately skips the md5/integrity plumbing of DataSource.
            cast(
                DataSource,
                StubDataSource(
                    split_skill_sequence_data=split_frame,
                    metadata=metadata,
                ),
            )
        )

    return _make


# --- event-annotated split frames ---------------------------------------------

_EVENT_METADATA = {
    "max_question_seq_len": 4,
    "num_questions": 1,
    "num_skills": 3,
    "kfold_n_splits": 2,
    "num_users": 3,
}


def event_frame() -> pl.DataFrame:
    """Three single-user sequences of three events each.

    Event 2 of every sequence repeats question 0 but annotates a *different*
    skill list, so a packer that reconstructs skills from the question column
    instead of reading the observed list is caught immediately.
    """
    return pl.DataFrame(
        {
            "sequence_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "seq_pos": [0, 1, 2] * 3,
            "user": [0] * 3 + [1] * 3 + [2] * 3,
            "question": [0] * 9,
            "interaction_id": list(range(9)),
            "label": [1, 0, 1] * 3,
            "timestamp": list(range(9)),
            "ms_first_response": [1000] * 9,
            "attempt_count": [1] * 9,
            "hint_count": [0] * 9,
            "fold": [0] * 3 + [1] * 3 + [-1] * 3,
            "skills": pl.Series([[0], [1], [0, 2]] * 3, dtype=pl.List(pl.Int32)),
        }
    )


def event_skill_frame() -> pl.DataFrame:
    """The event frame flattened to one scalar skill per interaction."""
    return (
        event_frame()
        .with_columns(pl.col("skills").list.first().alias("skill"))
        .drop("skills")
    )


class EventSource:
    """Duck-typed DataSource exposing the event frame in memory.

    ``loads`` counts ``get_split_question_sequence_data`` calls so tests can
    assert the loader did (or did not) touch the source.
    """

    dataset = "events"

    def __init__(self) -> None:
        self.loads = 0
        self.data = event_frame()

    def get_split_question_sequence_data(self, **kwargs: Any) -> pl.DataFrame:
        self.loads += 1
        return self.data

    def get_metadata(self, key: str | None = None) -> Any:
        return _EVENT_METADATA if key is None else _EVENT_METADATA[key]

    def get_relation(self, name: str) -> pl.DataFrame:
        raise AssertionError("Observed sequence skills must not use item relations")


class EventQuestionData(QuestionModelData):
    def prepare_data(self, args: Any) -> None: ...


class EventSkillData(SkillModelData):
    def prepare_data(self, args: Any) -> None: ...


class ParquetSource(DataSource):
    """DataSource whose event frame is materialized on disk (real md5 path)."""

    def load_src_data(self) -> None: ...

    def clean_raw_data(self) -> None: ...

    def transform_data(self) -> None: ...


class ParquetReads:
    """``pl.read_parquet`` stand-in recording the projected column sets."""

    def __init__(self) -> None:
        self.columns: list[tuple[str, ...]] = []
        self.read = pl.read_parquet

    def __call__(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        self.columns.append(tuple(kwargs["columns"]))
        return self.read(*args, **kwargs)


@pytest.fixture
def event_source() -> EventSource:
    return EventSource()


@pytest.fixture
def parquet_source(tmp_path: Path) -> ParquetSource:
    source = ParquetSource("events", str(tmp_path))
    Path(source.data_folder).mkdir()
    path = Path(source.data_folder) / "events_split_question_sequence.parquet"
    event_frame().write_parquet(path)
    source.metadata.update(_EVENT_METADATA)
    source.metadata["split_question_sequence_data_md5"] = source.compute_md5(str(path))
    return source


@pytest.fixture
def parquet_reads(monkeypatch: pytest.MonkeyPatch) -> ParquetReads:
    """Install the read recorder for the duration of one test."""
    reads = ParquetReads()
    monkeypatch.setattr(pl, "read_parquet", reads)
    return reads
