"""Tests for DataLoader worker configuration."""

import os

from utils.config import DataLoaderConfig


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
