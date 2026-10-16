import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import TensorDataset

from utils.data_process import DataSource
from utils.model_data import BaseModelData, QuestionModelData, SkillModelData
from utils.model_data.packed_sequences import (
    to_first_skill_ids,
    to_multi_skill_ids,
    to_skill_combination_ids,
)


class _Source:
    dataset = "events"

    def __init__(self) -> None:
        self.loads = 0
        self.data = pl.DataFrame(
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

    def get_split_question_sequence_data(self, **kwargs: Any) -> pl.DataFrame:
        self.loads += 1
        return self.data

    def get_metadata(self, key: str | None = None) -> Any:
        metadata = {
            "max_question_seq_len": 4,
            "num_questions": 1,
            "num_skills": 3,
            "kfold_n_splits": 2,
            "num_users": 3,
        }
        return metadata if key is None else metadata[key]

    def get_relation(self, name: str) -> pl.DataFrame:
        raise AssertionError("Observed sequence skills must not use item relations")


class _Data(QuestionModelData):
    def prepare_data(self, args: Any) -> None: ...


class _SkillData(SkillModelData):
    def prepare_data(self, args: Any) -> None: ...


def test_annotation_packing_keeps_event_skills_for_repeated_question() -> None:
    source = _Source()
    data = _Data(cast(DataSource, source))
    sequences = data.load_sequences()
    packed = sequences.skills
    assert packed.shape == (3, 4, 2)
    assert packed[0].tolist() == [[0, -1], [1, -1], [0, 2], [-1, -1]]
    assert to_first_skill_ids(sequences.skills)[0].tolist() == [0, 1, 0, 0]
    assert to_first_skill_ids(sequences.skills, offset=1)[0].tolist() == [1, 2, 1, 0]
    assert to_multi_skill_ids(sequences.skills, padding_value=9, offset=1)[
        0
    ].tolist() == [
        [1, 9],
        [2, 9],
        [1, 3],
        [9, 9],
    ]
    assert source.loads == 1


def test_folding_uses_observed_combinations_with_zero_reserved_for_padding() -> None:
    source = _Source()
    folded, count = to_skill_combination_ids(
        _Data(cast(DataSource, source)).pack_sequences(source.data).skills
    )
    assert count == 3
    assert len(set(folded[0, :3])) == 3
    assert folded[:, -1].tolist() == [0, 0, 0]
    np.testing.assert_array_equal(folded[0], folded[1])


@pytest.mark.parametrize("offset", [0, 1])
def test_to_first_skill_ids_own_only_scalar_storage_and_keep_row_alignment(
    offset: int,
) -> None:
    source = _Source()
    source.data = source.data.reverse()
    sequences = _Data(cast(DataSource, source)).pack_sequences(source.data)
    primary = to_first_skill_ids(sequences.skills, offset=offset)

    np.testing.assert_array_equal(
        primary,
        np.tile([offset, 1 + offset, offset, 0], (3, 1)),
    )
    assert primary.dtype == np.int64
    assert primary.flags.owndata
    assert primary.flags.c_contiguous
    assert primary.nbytes == 3 * 4 * np.dtype(np.int64).itemsize


def test_packers_use_supplied_rows_without_loading_or_mutating_source() -> None:
    source = _Source()
    data = _Data(cast(DataSource, source))
    frame = source.data.head(2).reverse()
    original = frame.clone()

    sequences = data.pack_sequences(frame)
    questions, responses, mask = sequences.question, sequences.response, sequences.mask
    skills = to_multi_skill_ids(sequences.skills, padding_value=3)
    primary = to_first_skill_ids(sequences.skills, offset=1)
    folded, count = to_skill_combination_ids(sequences.skills)

    assert source.loads == 0
    assert frame.equals(original)
    assert questions.shape == responses.shape == mask.shape == (1, 4)
    assert skills.tolist() == [[[0], [1], [3], [3]]]
    assert primary.tolist() == [[1, 2, 0, 0]]
    assert folded.tolist() == [[1, 2, 0, 0]]
    assert count == 2


class _ParquetSource(DataSource):
    def load_src_data(self) -> None: ...

    def clean_raw_data(self) -> None: ...

    def transform_data(self) -> None: ...


@pytest.fixture
def parquet_source(tmp_path: Path) -> _ParquetSource:
    source = _ParquetSource("events", str(tmp_path))
    folder = Path(source.data_folder)
    folder.mkdir()
    path = folder / "events_split_question_sequence.parquet"
    _Source().data.write_parquet(path)
    source.metadata.update(_Source().get_metadata())
    source.metadata["split_question_sequence_data_md5"] = source.compute_md5(str(path))
    return source


class _ParquetReads:
    def __init__(self) -> None:
        self.columns: list[tuple[str, ...]] = []
        self.read = pl.read_parquet

    def __call__(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        self.columns.append(tuple(kwargs["columns"]))
        return self.read(*args, **kwargs)


@pytest.mark.parametrize(
    ("model_data_class", "max_seq_len"),
    [(_Data, 4), (_SkillData, 6)],
)
def test_load_sequences_matches_explicit_packing_and_reuses_projection(
    model_data_class: type[BaseModelData],
    max_seq_len: int,
    parquet_source: _ParquetSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if model_data_class is _SkillData:
        skill_frame = (
            _Source()
            .data.with_columns(pl.col("skills").list.first().alias("skill"))
            .drop("skills")
        )
        skill_frame.write_parquet(
            Path(parquet_source.data_folder) / "events_split_skill_sequence.parquet"
        )
        parquet_source.metadata["max_skill_seq_len"] = max_seq_len

    reads = _ParquetReads()
    monkeypatch.setattr(pl, "read_parquet", reads)
    data = model_data_class(parquet_source)
    required = ("timestamp",)
    optional = ("attempt_count", "absent_feature")

    sequences = data.load_sequences(required=required, optional=optional)
    frame = data.load_split_frame(required=required, optional=optional)
    explicit = data.pack_sequences(frame)

    assert len(reads.columns) == len(parquet_source._data_cache) == 1
    assert sequences.mask.shape == (3, max_seq_len)
    assert sequences.num_folds == explicit.num_folds == 2
    for field in (
        "question",
        "response",
        "mask",
        "skills",
        "sequence_id",
        "user",
        "fold",
    ):
        actual = getattr(sequences, field)
        expected = getattr(explicit, field)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    assert (
        set(sequences.features)
        == set(explicit.features)
        == {
            "interaction_id",
            "timestamp",
            "attempt_count",
        }
    )
    for name in sequences.features:
        np.testing.assert_array_equal(sequences.features[name], explicit.features[name])
    assert sequences.partition(0).train.tolist() == [1]
    assert sequences.partition(0).val.tolist() == [0]
    assert sequences.partition(0).test.tolist() == [2]


def test_load_sequences_rejects_missing_required_feature_before_reading(
    parquet_source: _ParquetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _ParquetReads()
    monkeypatch.setattr(pl, "read_parquet", reads)
    with pytest.raises(ValueError, match=r"missing required column.*absent_feature"):
        _Data(parquet_source).load_sequences(required=("absent_feature",))
    assert reads.columns == []


def test_packers_and_fold_split_reuse_one_real_projection(
    parquet_source: _ParquetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _ParquetReads()
    monkeypatch.setattr(pl, "read_parquet", reads)
    data = _Data(parquet_source)
    frame = data.load_split_frame(required=("timestamp",))

    sequences = data.pack_sequences(frame)
    dataset = TensorDataset(
        torch.from_numpy(sequences.question),
        torch.from_numpy(sequences.skills),
    )
    train, val, test = sequences.partition(0).apply(dataset)

    assert len(reads.columns) == len(parquet_source._data_cache) == 1
    assert "timestamp" in reads.columns[0]
    assert data.load_split_frame(required=("timestamp",)) is frame
    assert len(reads.columns) == 1
    assert train.dataset is val.dataset is test.dataset is dataset
    for split in (train, val, test):
        assert len(split) == 1
        np.testing.assert_array_equal(split[0][1][:3], [[0, -1], [1, -1], [0, 2]])


@pytest.mark.parametrize(
    "name",
    [
        "QIKT",
        "IEKT",
        "PSKT",
        "DKT2",
        "DisKT",
        "ReKT",
        "HDKT",
        "TCKT",
        "LBKT",
        "LPKT",
        "MIKT",
        "GRKT",
        "ClusterKT",
        "HiTSKT",
    ],
)
def test_model_adapters_read_one_split_projection(
    name: str, parquet_source: _ParquetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _ParquetReads()
    monkeypatch.setattr(pl, "read_parquet", reads)
    module = importlib.import_module(f"model.{name}.{name}_data")
    model_data = getattr(module, f"{name}ModelData")(parquet_source)
    model_data.prepare_data(
        SimpleNamespace(
            data=SimpleNamespace(fold=0),
            model=SimpleNamespace(
                max_it_minutes=120,
                max_rt_seconds=300,
                q_gamma=0.1,
                n_et=300,
                action_size=4,
                session_size=2,
                timestamp_unit="seconds",
                session_gap_hours=1,
            ),
        )
    )

    assert len(reads.columns) == len(parquet_source._data_cache) == 1


@pytest.mark.parametrize(
    "name",
    ["DKT", "AKT", "SAKT", "DKVMN", "DeepIRT", "DKTForget", "FAKT", "MCSKT", "MTKT"],
)
def test_skill_adapters_pack_and_partition_one_projection(
    name: str, parquet_source: _ParquetSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = Path(parquet_source.data_folder)
    data = _Source().data.with_columns(pl.col("skills").list.first().alias("skill"))
    data.write_parquet(folder / "events_split_skill_sequence.parquet")
    windows = data.head(3).select(
        pl.lit(0).alias("sample_id"),
        pl.col("seq_pos").alias("position"),
        "skill",
        "question",
        pl.col("label").alias("response"),
        (pl.col("seq_pos") == 2).cast(pl.Int8).alias("mask"),
        pl.col("user").alias("user_id"),
        pl.col("interaction_id").alias("group_id"),
        pl.col("label").alias("true_label"),
        "timestamp",
    )
    windows.write_parquet(folder / "events_windowlate.parquet")
    parquet_source.metadata.update(max_skill_seq_len=4, max_windowlate_seq_len=4)
    reads = _ParquetReads()
    monkeypatch.setattr(pl, "read_parquet", reads)
    module = importlib.import_module(f"model.{name}.{name}_data")
    model_data = getattr(module, f"{name}ModelData")(parquet_source)
    train, val, _ = model_data.prepare_data(
        SimpleNamespace(
            data=SimpleNamespace(fold=0),
            general=SimpleNamespace(device="cpu", pin_memory=False),
            model=SimpleNamespace(batch_size=2, num_rgap=16, num_sgap=16, num_pcount=8),
        )
    )
    assert len(reads.columns) == len(parquet_source._data_cache) == 1
    assert train.dataset is val.dataset
    assert train.indices == [1]
    assert val.indices == [0]


def test_skills_to_multihot_keeps_event_boundaries_and_excludes_padding() -> None:
    import torch

    from model.layers.interaction import skills_to_multihot

    skills = torch.tensor([[[0, 3], [1, 2], [3, 3]]])
    assert skills_to_multihot(skills, num_skills=3).tolist() == [
        [[True, False, False], [False, True, True], [False, False, False]]
    ]


@pytest.mark.parametrize(
    "name", ["QIKT", "IEKT", "PSKT", "DKT2", "ReKT", "HDKT", "TCKT", "LBKT", "LPKT"]
)
def test_model_adapters_do_not_reconstruct_sequence_skills_from_questions(
    name: str,
) -> None:
    module = importlib.import_module(f"model.{name}.{name}_data")
    cls = getattr(module, f"{name}ModelData" if name != "ReKT" else "ReKTModelData")
    prepared = cls(_Source()).prepare_data(
        SimpleNamespace(
            data=SimpleNamespace(fold=0),
            model=SimpleNamespace(max_it_minutes=120, max_rt_seconds=300, q_gamma=0.1),
        )
    )
    train = prepared[0]
    assert len(train) == 1
    assert prepared[0].dataset is prepared[1].dataset is prepared[2].dataset
    first = np.asarray(train.dataset.skills)[train.indices[0]]
    assert not np.array_equal(first[0], first[1])


def test_axis_packing_keeps_distinct_skills_for_steps_of_same_question() -> None:
    import torch

    from model.AxisKT.AxisKT_data import AxisKTDataset, axiskt_packed_collate_fn

    questions = np.zeros((1, 3), dtype=np.int64)
    dataset = AxisKTDataset(
        questions,
        questions,
        questions,
        np.ones_like(questions),
        np.array([[[0], [1], [0]]]),
        np.ones((1, 3, 1), dtype=bool),
    )
    batch = axiskt_packed_collate_fn([dataset[0]])
    assert batch[-2].tolist() == [[[0], [1], [0]]]
    assert batch[4].tolist() == [[0, 2, 1]]
    assert torch.equal(batch[-1], torch.ones((1, 3, 1), dtype=torch.bool))
