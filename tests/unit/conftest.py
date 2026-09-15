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

import pytest


@pytest.fixture
def registry_snapshot():
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
def isolated_loggers():
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
def clean_log_level_env(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)


@pytest.fixture(scope="session")
def tiny_model_config_name():
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
def make_run_config(tiny_model_config_name):
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
        log_dir=None,
        epochs=1,
        batch_size=2,
        skip_test=True,
        cloud_tracking=False,
        save_last_checkpoint=False,
        checkpoint_path=None,
        early_stopping=None,
        device="cpu",
        seed=42,
        dataset="tinyds",
        log_batch_metrics=False,
        pin_memory=None,
        progress="none",
        model_kwargs=None,
    ):
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
