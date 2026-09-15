"""Shared fixtures for case analysis tests: a minimal analyzer stack."""

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import torch
from torch.utils.data import Dataset

from utils.case_analysis.base_analyzer import BaseCaseAnalyzer
from utils.training.runtime_components import RuntimeComponents


class _RC:
    """Minimal RunConfig stand-in with the nodes the base class reads."""

    class general:
        device = "cpu"
        seed = None
        deterministic = True

    class model:
        batch_size = 2

    class experiment:
        model_name = "DummyModel"

    class data:
        dataset = "dummyds"


class SeqDataset(Dataset):
    """Four single-user samples, 3 positions each."""

    def __len__(self) -> int:
        return 4

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(idx),
            torch.tensor([idx, idx + 1, idx + 2]),
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 1, 1]),
        )


class DummyAnalyzer(BaseCaseAnalyzer):
    """Analyzer whose model outputs the sequence sum per position."""

    def build_components(self, rc: Any, data_src: Any) -> RuntimeComponents:
        self.extra_tensor = torch.zeros(2)
        return RuntimeComponents(model=torch.nn.Linear(3, 1), val_data=SeqDataset())

    def on_device(self, device: torch.device) -> None:
        self.extra_tensor = self.extra_tensor.to(device)

    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        _users, sequences, responses, _masks = batch_data
        logits = self.model(sequences.float())
        y_hat = logits.expand_as(responses.float()).reshape(-1)
        y_label = responses.reshape(-1).float()
        y_predict = self._generate_binary_predictions(y_hat)
        return {"y_hat": y_hat, "y_label": y_label, "y_predict": y_predict}

    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        users, sequences, _responses, _masks = batch_data
        users_rep = users.repeat_interleave(sequences.shape[1])
        return {
            "user_ids": users_rep.tolist(),
            "question_ids": sequences.reshape(-1).tolist(),
            "labels": outputs["y_label"].tolist(),
            "predictions": outputs["y_predict"].tolist(),
        }


@pytest.fixture
def rc() -> _RC:
    return _RC()


@pytest.fixture
def dummy_analyzer_cls() -> type[DummyAnalyzer]:
    return DummyAnalyzer


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> str:
    p = tmp_path / "dummy_model.pth"
    torch.save(torch.nn.Linear(3, 1).state_dict(), p)
    return str(p)
