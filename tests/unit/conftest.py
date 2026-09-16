"""Cross-area unit-test fixtures: global-state isolation and config factories.

Shared by the core/config/training/optuna_utils/data_process areas so each
gets one implementation. Anti-pollution rules:

- Any test that registers, indexes, clears, or discovers registry state must
  request ``registry_snapshot``.
- Any test touching ``add_file_handler``/``set_log_level`` must request
  ``isolated_loggers`` (plus ``clean_log_level_env`` for env-reading paths).
- Filesystem-touching defaults must stay inside ``tmp_path``: pass explicit
  ``base_dir``/``data_base_path``/``log_dir`` — never rely on ``runs/`` or
  ``./data`` defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from utils.config.run_config import EarlyStoppingConfig, RunConfig


@pytest.fixture
def registry_snapshot() -> Iterator[None]:
    """Snapshot every global registry (and the class-level roll-call) and
    restore both tables exactly on teardown."""
    from utils.core.registry import UniversalRegistry

    roll_call = list(UniversalRegistry._all_registries)
    saved = [(r, dict(r._registry), dict(r._index)) for r in roll_call]
    yield
    UniversalRegistry._all_registries[:] = roll_call
    for r, registry_table, index_table in saved:
        r._registry.clear()
        r._registry.update(registry_table)
        r._index.clear()
        r._index.update(index_table)


@pytest.fixture
def isolated_loggers() -> Iterator[ModuleType]:
    """Empty the logger cache for the test; strip handlers from every logger
    created during it, then restore the original cache."""
    from utils.core import logger as logger_module

    saved = dict(logger_module._loggers)
    logger_module._loggers.clear()
    yield logger_module
    for lg in logger_module._loggers.values():
        for h in list(lg.handlers):
            lg.removeHandler(h)
            h.close()
    logger_module.reset_loggers()
    logger_module._loggers.clear()
    logger_module._loggers.update(saved)


@pytest.fixture
def clean_log_level_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)


@pytest.fixture(scope="session")
def tiny_model_config_name() -> Iterator[str]:
    """Register a throwaway concrete ModelConfig for the whole session.

    Inserted directly into ``MODEL_CONFIGS._registry`` so repeated sessions
    cannot trip the duplicate-class KeyError from ``register()``. Gives the
    config parser / archive / optuna tests a model node without importing
    the real training stack.
    """
    from dataclasses import dataclass

    from utils.config.run_config import ModelConfig
    from utils.core import MODEL_CONFIGS

    @dataclass
    class TinyTestModelConfig(ModelConfig):
        epochs: int = 2
        batch_size: int = 4
        hidden_dim: int = 8
        dropout: float = 0.1

    MODEL_CONFIGS._registry["TinyTestModel"] = TinyTestModelConfig
    yield "TinyTestModel"
    MODEL_CONFIGS._registry.pop("TinyTestModel", None)


@pytest.fixture
def make_run_archive(
    tmp_path: Path, tiny_model_config_name: str
) -> Callable[..., Path]:
    """Factory: fake run directory carrying a minimal run_config.yaml archive.

    Only the model name and dataset are written — every other node falls back
    to its dataclass default, exactly like a sparse real archive would.
    """

    def _make(overrides: dict[str, Any] | None = None) -> Path:
        import yaml

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "experiment": {"model_name": tiny_model_config_name},
            "data": {"dataset": "tinyds"},
        }
        for dotted, value in (overrides or {}).items():
            node, _, field = dotted.partition(".")
            data.setdefault(node, {})[field] = value
        (run_dir / "run_config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        return run_dir

    return _make


@pytest.fixture
def make_run_config(tiny_model_config_name: str) -> Callable[..., RunConfig]:
    """Factory for a real RunConfig wired for fast offline tests:
    cloud tracking off, test-skip on, last-checkpoint off, CPU device."""
    from utils.config.run_config import (
        EarlyStoppingConfig,
        ExperimentConfig,
        GeneralConfig,
        RunConfig,
        RunDataConfig,
    )
    from utils.core import MODEL_CONFIGS

    def _make(
        *,
        log_dir: str | None = None,
        epochs: int = 1,
        batch_size: int = 2,
        skip_test: bool = True,
        cloud_tracking: bool = False,
        save_last_checkpoint: bool = False,
        checkpoint_path: str | None = None,
        early_stopping: EarlyStoppingConfig | None = None,
        device: str = "cpu",
        seed: int = 42,
        dataset: str = "tinyds",
        log_batch_metrics: bool = False,
        pin_memory: bool | None = None,
        progress: Literal["auto", "rich", "none"] = "none",
        model_kwargs: dict[str, Any] | None = None,
    ) -> RunConfig:
        model_cls = MODEL_CONFIGS._registry[tiny_model_config_name]
        return RunConfig(
            general=GeneralConfig(
                log_dir=log_dir,
                checkpoint_path=checkpoint_path,
                device=device,
                seed=seed,
                cloud_tracking=cloud_tracking,
                log_batch_metrics=log_batch_metrics,
                skip_test=skip_test,
                save_last_checkpoint=save_last_checkpoint,
                pin_memory=pin_memory,
                progress=progress,
            ),
            early_stopping=early_stopping or EarlyStoppingConfig(patience=2),
            experiment=ExperimentConfig(model_name=tiny_model_config_name),
            data=RunDataConfig(dataset=dataset),
            model=model_cls(**(model_kwargs or {})),
        )

    return _make
