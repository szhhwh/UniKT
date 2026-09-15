"""Tests for seed_everything: seeding coverage, deterministic flags, env writes."""

import os
import random

import numpy as np
import pytest
import torch

from utils.core.random import seed_everything


def _restore_env(key: str) -> None:
    saved = os.environ.get(key)
    if saved is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = saved


class TestSeeding:
    def test_none_seed_returns_none_and_untouches_random_state(self) -> None:
        state = random.getstate()
        assert seed_everything(None) is None
        assert random.getstate() == state

    def test_returns_the_seed(self) -> None:
        assert seed_everything(123) == 123

    def test_reproducible_random_and_numpy(self) -> None:
        seed_everything(42)
        py_seq = [random.random() for _ in range(3)]
        np_seq = list(np.random.random(3))

        seed_everything(42)
        assert [random.random() for _ in range(3)] == py_seq
        assert list(np.random.random(3)) == np_seq

    def test_reproducible_torch(self) -> None:
        seed_everything(7)
        first = torch.rand(4)
        seed_everything(7)
        assert torch.equal(first, torch.rand(4))

    def test_writes_pythonhashseed_env(self) -> None:
        try:
            seed_everything(99)
            assert os.environ["PYTHONHASHSEED"] == "99"
        finally:
            _restore_env("PYTHONHASHSEED")

    def test_cuda_branch_guarded_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert seed_everything(5) == 5  # must not touch cuda seeds


class TestDeterministic:
    def test_true_enables_deterministic_algorithms_and_env(self) -> None:
        was_enabled = torch.are_deterministic_algorithms_enabled()
        try:
            seed_everything(11, deterministic=True)
            assert torch.are_deterministic_algorithms_enabled()
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
        finally:
            torch.use_deterministic_algorithms(was_enabled)
            _restore_env("CUBLAS_WORKSPACE_CONFIG")

    def test_false_leaves_flags_and_env_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        torch.use_deterministic_algorithms(False)
        seed_everything(13, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

    def test_polars_branch_guarded(self) -> None:
        # Polars may or may not be installed; either way the call must succeed.
        seed_everything(21, deterministic=False)
