"""Equivalence of raw-view batch construction and legacy windowlate loading."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from model.AKT.AKT_model import AKT
from model.DeepIRT.DeepIRT_data import DeepIRTModelData
from model.DKT.DKT_model import DKT
from model.DKTForget.DKTForget_data import (
    DKTForgetModelData,
    DKTForgetWindowlateIterableDataset,
)
from model.FAKT.FAKT_data import FAKTModelData, FAKTWindowlateIterableDataset
from model.MCSKT.MCSKT_data import MCSKTModelData, MCSKTWindowlateIterableDataset
from model.MTKT.MTKT_data import MTKTModelData, MTKTWindowlateIterableDataset
from utils.config.data_config import DataLoaderConfig, create_optimized_dataloader
from utils.model_data.skill_model_data import SkillModelData, WindowlateIterableDataset
from utils.training.metrics import MetricsAccumulator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def windowlate_path(tmp_path: Path) -> str:
    # Multi-skill targets share histories/group ids; lengths include singleton,
    # short, full and expanded windows. Row groups contain 2/1/3/3 samples.
    lengths = [1, 2, 2, 4, 4, 6, 6, 8, 8]
    groups = [101, 205, 205, 309, 309, 411, 411, 512, 610]
    labels = [1, 0, 0, 1, 1, 0, 0, 1, 1]
    samples = []
    for index, (length, group, label) in enumerate(zip(lengths, groups, labels)):
        positions = np.arange(length)
        skill = (positions % 3 + 1).astype(np.int32)
        skill[-1] = index % 7 + 1
        response = (positions % 2).astype(np.int8)
        response[-1] = 0
        true_label = response.copy()
        true_label[-1] = label
        samples.append(
            pa.table(
                {
                    "sample_id": np.full(length, index * 3 + 11, dtype=np.int64),
                    "position": positions.astype(np.int32),
                    "skill": skill,
                    "question": (positions + 1).astype(np.int32),
                    "response": response,
                    "mask": (positions == length - 1).astype(np.int8),
                    "user_id": np.full(length, 7 if index < 5 else 42, dtype=np.int32),
                    "group_id": np.full(length, group, dtype=np.int64),
                    "true_label": true_label,
                    "timestamp": (positions * 60_000 + 1_000).astype(np.int64),
                }
            )
        )
    path = str(tmp_path / "stub_windowlate.parquet")
    with pq.ParquetWriter(path, samples[0].schema) as writer:
        for start, end in ((0, 2), (2, 3), (3, 6), (6, 9)):
            writer.write_table(pa.concat_tables(samples[start:end]))
    return path


@pytest.fixture
def windowlate_model_data(
    windowlate_path: str, make_skill_model_data: Callable[..., SkillModelData]
) -> SkillModelData:
    from pathlib import Path

    model_data = make_skill_model_data(
        metadata={
            "max_windowlate_seq_len": 8,
            "max_skill_seq_len": 8,
            "kfold_n_splits": 2,
        }
    )
    model_data.data_src.data_folder = str(Path(windowlate_path).parent)
    return model_data


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("drop_last", [False, True])
def test_batches_match_legacy_loader(
    windowlate_model_data: SkillModelData,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
) -> None:
    dataset = windowlate_model_data._create_windowlate_dataset()
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": drop_last,
    }
    legacy = list(DataLoader(dataset, **kwargs))
    optimized_loader = windowlate_model_data.create_windowlate_dataloader(**kwargs)
    optimized = list(optimized_loader)
    assert len(optimized_loader) == len(DataLoader(dataset, **kwargs))
    assert len(optimized) == len(legacy)
    for actual, expected in zip(optimized, legacy, strict=True):
        assert type(actual) is type(expected)
        for actual_field, expected_field in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)
    # Constructing a loader must not switch the source's public iterator to raw views.
    assert isinstance(next(iter(dataset)), tuple)


def test_batch_path_skips_single_sample_tensor_builder(
    windowlate_model_data: SkillModelData, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise AssertionError("The batched path must not allocate per-sample tensors")

    monkeypatch.setattr(WindowlateIterableDataset, "_build_single_tensor", fail)
    batch = next(iter(windowlate_model_data.create_windowlate_dataloader(batch_size=4)))
    assert len(batch) == 7
    assert batch[0].shape == (4, 8)


@pytest.mark.parametrize(
    ("model_data_class", "dataset_class"),
    [
        (DKTForgetModelData, DKTForgetWindowlateIterableDataset),
        (FAKTModelData, FAKTWindowlateIterableDataset),
        (MCSKTModelData, MCSKTWindowlateIterableDataset),
        (MTKTModelData, MTKTWindowlateIterableDataset),
    ],
)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_custom_window_features_match_legacy_loader(
    windowlate_path: str,
    windowlate_model_data: SkillModelData,
    model_data_class: type,
    dataset_class: type,
    num_workers: int,
) -> None:
    dataset = dataset_class(
        windowlate_path, max_seq_len=8, num_rgap=16, num_sgap=16, num_pcount=8
    )
    model_data = model_data_class(windowlate_model_data.data_src)
    model_data.num_rgap, model_data.num_sgap, model_data.num_pcount = 16, 16, 8
    assert isinstance(model_data._create_windowlate_dataset(), dataset_class)
    kwargs: dict[str, Any] = {"batch_size": 4, "num_workers": num_workers}
    legacy = list(DataLoader(dataset, **kwargs))
    loaders = (
        model_data.create_windowlate_dataloader(**kwargs),
        create_optimized_dataloader(
            dataset,
            batch_size=4,
            shuffle=False,
            config=DataLoaderConfig(num_workers=num_workers, persistent_workers=False),
            device=torch.device("cpu"),
        ),
    )
    for loader in loaders:
        optimized = list(loader)
        assert len(optimized) == len(legacy)
        for actual, expected in zip(optimized, legacy, strict=True):
            assert len(actual) == len(expected) == 9
            for actual_field, expected_field in zip(actual, expected, strict=True):
                torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_public_loader_options(
    windowlate_model_data: SkillModelData, num_workers: int
) -> None:
    loader = windowlate_model_data.create_windowlate_dataloader(
        batch_size=4,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        drop_last=True,
        timeout=0,
    )
    assert loader.batch_size == 4
    assert loader.num_workers == num_workers
    assert loader.pin_memory is True
    assert loader.drop_last is True
    assert loader.prefetch_factor == (3 if num_workers else None)
    assert loader.persistent_workers is bool(num_workers)


def _training_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sequence_id": np.repeat(np.arange(4), 8),
            "seq_pos": np.tile(np.arange(8), 4),
            "user": np.repeat(np.arange(4), 8),
            "fold": np.repeat([1, 1, 0, 0], 8),
            "question": [0] * 32,
            "skill": [1] * 32,
            "label": [0] * 32,
            "timestamp": np.tile(np.arange(8) * 60_000, 4),
        }
    )


@pytest.mark.parametrize("num_workers", [0, 2])
def test_deepirt_prepare_preserves_test_loader_options(
    windowlate_model_data: SkillModelData,
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
) -> None:
    model_data = DeepIRTModelData(windowlate_model_data.data_src)
    monkeypatch.setattr(
        model_data, "load_split_frame", lambda **kwargs: _training_frame()
    )
    rc = SimpleNamespace(
        data=SimpleNamespace(fold=0),
        model=SimpleNamespace(
            batch_size=4,
            test_batch_size=2,
            test_num_workers=num_workers,
            test_pin_memory=False,
            test_prefetch_factor=3,
        ),
    )
    train, val, loader = model_data.prepare_data(rc)
    assert len(train) == len(val) == 2
    assert loader.batch_size == 2
    assert loader.num_workers == num_workers
    assert loader.pin_memory is False
    assert loader.prefetch_factor == (3 if num_workers else None)
    assert next(iter(loader))[0].shape == (2, 8)


@pytest.mark.parametrize("model_data_class", [MCSKTModelData, MTKTModelData])
def test_prepare_supplies_time_feature_hook_configuration(
    windowlate_model_data: SkillModelData,
    monkeypatch: pytest.MonkeyPatch,
    model_data_class: type,
) -> None:
    model_data = model_data_class(windowlate_model_data.data_src)
    monkeypatch.setattr(
        model_data, "load_split_frame", lambda **kwargs: _training_frame()
    )
    rc = SimpleNamespace(
        data=SimpleNamespace(fold=0),
        model=SimpleNamespace(num_rgap=16, num_sgap=16, num_pcount=8),
    )
    _, _, dataset = model_data.prepare_data(rc)
    assert (dataset.num_rgap, dataset.num_sgap, dataset.num_pcount) == (16, 16, 8)
    loader = model_data.create_windowlate_dataloader(batch_size=4)
    for actual, expected in zip(loader, DataLoader(dataset, batch_size=4), strict=True):
        for actual_field, expected_field in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)


def test_shared_factory_uses_batch_construction(windowlate_path: str) -> None:
    dataset = WindowlateIterableDataset(windowlate_path, max_seq_len=8)
    loader = create_optimized_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        config=DataLoaderConfig(num_workers=0),
        device=torch.device("cpu"),
    )
    assert loader.collate_fn == dataset._collate_batch
    assert next(iter(loader))[0].shape == (4, 8)


def test_spawn_workers_match_legacy_loader(
    windowlate_model_data: SkillModelData,
) -> None:
    dataset = windowlate_model_data._create_windowlate_dataset()
    kwargs: dict[str, Any] = {
        "batch_size": 4,
        "num_workers": 2,
        "multiprocessing_context": "spawn",
    }
    for actual, expected in zip(
        windowlate_model_data.create_windowlate_dataloader(**kwargs),
        DataLoader(dataset, **kwargs),
        strict=True,
    ):
        for actual_field, expected_field in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)


def test_persistent_workers_preserve_repeated_iteration(
    windowlate_model_data: SkillModelData,
) -> None:
    dataset = windowlate_model_data._create_windowlate_dataset()
    kwargs: dict[str, Any] = {
        "batch_size": 4,
        "num_workers": 2,
        "persistent_workers": True,
    }
    legacy = DataLoader(dataset, **kwargs)
    optimized = windowlate_model_data.create_windowlate_dataloader(**kwargs)
    for _ in range(2):
        for actual, expected in zip(optimized, legacy, strict=True):
            for actual_field, expected_field in zip(actual, expected, strict=True):
                torch.testing.assert_close(actual_field, expected_field, rtol=0, atol=0)


@pytest.mark.parametrize("model_name", ["dkt", "akt"])
def test_predictions_and_group_metrics_are_identical(
    windowlate_model_data: SkillModelData, model_name: str
) -> None:
    dataset = windowlate_model_data._create_windowlate_dataset()
    torch.manual_seed(7)
    model: torch.nn.Module
    if model_name == "dkt":
        model = DKT(num_c=10, emb_size=8, dropout=0).eval()
    else:
        model = AKT(
            num_c=10,
            d_model=8,
            n_blocks=1,
            num_attn_heads=2,
            d_ff=16,
            final_fc_dim=16,
            dropout=0,
        ).eval()

    def evaluate(loader: DataLoader) -> tuple[torch.Tensor, dict[str, float]]:
        accumulator = MetricsAccumulator()
        predictions = []
        with torch.inference_mode():
            for sequence, response, mask, group_id, labels, *_ in loader:
                if model_name == "dkt":
                    full_scores = model(sequence, response, mask)[:, 1:]
                    mask, labels, group_id = mask[:, 1:], labels[:, 1:], group_id[:, 1:]
                else:
                    full_scores, _ = model(sequence, response, mask)
                scores = full_scores[mask]
                predictions.append(scores)
                accumulator.update(
                    "test",
                    {
                        "y_label": labels[mask].float(),
                        "y_score": scores,
                        "y_prob": scores,
                        "y_predict": (scores >= 0.5).int(),
                        "group_id": group_id[mask],
                    },
                )
        return torch.cat(predictions), accumulator.compute("test")

    legacy_predictions, legacy_metrics = evaluate(DataLoader(dataset, batch_size=4))
    predictions, metrics = evaluate(
        windowlate_model_data.create_windowlate_dataloader(batch_size=4)
    )
    torch.testing.assert_close(predictions, legacy_predictions, rtol=0, atol=0)
    assert "mean_auc" in metrics
    assert metrics == legacy_metrics
