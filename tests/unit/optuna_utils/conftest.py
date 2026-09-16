"""Shared fixtures for optuna_utils tests: duck-typed trials and yaml factories.

Anti-pollution rules:
- Any test registering throwaway model configs must request
  ``registry_snapshot`` (from ``tests/unit/conftest.py``).
- All yaml factories write inside ``tmp_path``; studies stay in-memory
  (``db_url`` unset, ``save_dir`` only under ``tmp_path``).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from optuna.trial import FixedTrial


class _FakeTrial:
    """Duck-typed optuna Trial: records report()/user attrs, toggleable pruning."""

    def __init__(self, should_prune: bool = False, number: int = 0) -> None:
        self._should_prune = should_prune
        self.number = number
        self.reports: list[tuple[float, int]] = []
        self.user_attrs: dict = {}

    def should_prune(self) -> bool:
        return self._should_prune

    def report(self, value: float, step: int) -> None:
        self.reports.append((float(value), int(step)))

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value


# Duck-typed trial doubles cross optuna's strictly-typed API (Trial and
# FixedTrial are sibling classes, not one hierarchy), so the fixtures below
# expose ``Any`` products; the factories themselves stay precisely typed.
@pytest.fixture
def fake_trial() -> Any:
    """A fresh recording trial double."""
    return _FakeTrial()


@pytest.fixture
def make_fake_trial() -> Callable[..., Any]:
    """Factory for recording trial doubles with pruning/number preconfigured."""

    def _make(should_prune: bool = False, number: int = 0) -> _FakeTrial:
        return _FakeTrial(should_prune=should_prune, number=number)

    return _make


@pytest.fixture
def fixed_trial() -> Callable[[dict[str, Any]], Any]:
    """Factory wrapping ``optuna.trial.FixedTrial`` over a params dict."""

    def _make(params: dict[str, Any]) -> FixedTrial:
        return FixedTrial(params)

    return _make


@pytest.fixture
def write_optuna_yaml(tmp_path: Path) -> Callable[..., str]:
    """Factory dumping a dict to yaml inside tmp_path; returns the path."""

    def _write(data: dict[str, Any], name: str = "optuna.yaml") -> str:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return str(path)

    return _write
