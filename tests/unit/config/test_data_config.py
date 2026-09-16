"""Tests for DataLoaderConfig and the optimized DataLoader factory."""

import os

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from utils.config.data_config import DataLoaderConfig, create_optimized_dataloader


class _ScalarDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([float(idx)])


class TestGetNumWorkers:
    def test_auto_caps_at_cpu_count(self) -> None:
        cfg = DataLoaderConfig(num_workers="auto")
        assert cfg.get_num_workers() == min(os.cpu_count() or 1, 8)

    def test_auto_respects_custom_limit(self) -> None:
        cfg = DataLoaderConfig(num_workers="auto")
        assert cfg.get_num_workers(max_limit=2) == min(os.cpu_count() or 1, 2)

    def test_int_passthrough(self) -> None:
        assert DataLoaderConfig(num_workers=4).get_num_workers() == 4

    def test_zero_and_over_limit_passthrough(self) -> None:
        assert DataLoaderConfig(num_workers=0).get_num_workers() == 0
        assert DataLoaderConfig(num_workers=99).get_num_workers() == 99


class TestCreateOptimizedDataloader:
    def test_missing_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Device information is required"):
            create_optimized_dataloader(_ScalarDataset(), device=None)

    def test_cpu_device_forces_pin_memory_off(self) -> None:
        loader = create_optimized_dataloader(
            _ScalarDataset(), device=torch.device("cpu")
        )
        assert loader.pin_memory is False

    def test_cuda_like_device_enables_pin_memory(self) -> None:
        fake_cuda = type("FakeCuda", (), {"type": "cuda"})()
        loader = create_optimized_dataloader(
            _ScalarDataset(), device=fake_cuda, num_workers=0
        )
        assert loader.pin_memory is True

    def test_pin_memory_param_overrides_auto(self) -> None:
        fake_cuda = type("FakeCuda", (), {"type": "cuda"})()
        loader = create_optimized_dataloader(
            _ScalarDataset(), device=fake_cuda, num_workers=0, pin_memory=False
        )
        assert loader.pin_memory is False

    def test_pin_memory_none_keeps_auto(self) -> None:
        fake_cuda = type("FakeCuda", (), {"type": "cuda"})()
        loader = create_optimized_dataloader(
            _ScalarDataset(), device=fake_cuda, num_workers=0, pin_memory=None
        )
        assert loader.pin_memory is True

    def test_pin_memory_param_cannot_pin_cpu(self) -> None:
        loader = create_optimized_dataloader(
            _ScalarDataset(), device=torch.device("cpu"), pin_memory=True
        )
        assert loader.pin_memory is False

    def test_zero_workers_normalizes_loader_args(self) -> None:
        loader = create_optimized_dataloader(
            _ScalarDataset(),
            config=DataLoaderConfig(num_workers=0),
            device=torch.device("cpu"),
        )
        assert loader.num_workers == 0
        assert loader.prefetch_factor is None
        assert loader.persistent_workers is False

    def test_zero_workers_via_kwargs_override_also_normalized(self) -> None:
        loader = create_optimized_dataloader(
            _ScalarDataset(), device=torch.device("cpu"), num_workers=0
        )
        assert loader.num_workers == 0
        assert loader.prefetch_factor is None
        assert loader.persistent_workers is False

    def test_kwargs_override_config(self) -> None:
        from torch.utils.data import SequentialSampler

        loader = create_optimized_dataloader(
            _ScalarDataset(),
            batch_size=3,
            shuffle=False,
            device=torch.device("cpu"),
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 3
        assert isinstance(loader.sampler, SequentialSampler)

    def test_returns_dataLoader_and_iterates(self) -> None:
        loader = create_optimized_dataloader(
            _ScalarDataset(), device=torch.device("cpu"), num_workers=0, batch_size=1
        )
        batches = list(loader)  # default shuffle=True, order not guaranteed
        assert len(batches) == 4
        assert sorted(b[0].item() for b in batches) == [0.0, 1.0, 2.0, 3.0]
