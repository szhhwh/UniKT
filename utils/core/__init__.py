"""Core infrastructure: unified registry, logger, and random seed utilities.

This module re-exports the primary components of the utils.core package,
including registries (trainers, model params, data sources, analyzers,
metric loggers), logging utilities, and seed setting.
"""

# isort: off
from .env import load_env
# isort: on

from .discovery import discover_registrations
from .logger import (
    add_file_handler,
    get_logger,
    remove_file_handler,
    reset_loggers,
    set_log_level,
)
from .random import seed_everything
from .registry import (
    ANALYZERS,
    CASE_SELECTORS,
    CASE_SINKS,
    CASE_VISUALIZERS,
    DATA_SOURCES,
    EFFICIENCY_STAGES,
    LLM_CLIENTS,
    METRIC_LOGGERS,
    METRICS,
    MODEL_CONFIGS,
    TRAINERS,
    UniversalRegistry,
    register_analyzer,
    register_case_selector,
    register_case_sink,
    register_case_visualizer,
    register_data_source,
    register_efficiency_stage,
    register_llm_client,
    register_metric,
    register_metric_logger,
    register_model_config,
    register_trainer,
)


def get_supported_datasets() -> list[str]:
    """Return all registered dataset names from ``DATA_SOURCES``.

    Lazily triggers discovery for ``utils.data_process`` if the registry
    is still empty (handles the case where ``utils.data_process`` was never
    imported, e.g. in the web backend).
    """
    if not DATA_SOURCES:
        from pathlib import Path

        discover_registrations(
            Path(__file__).resolve().parent.parent / "data_process",
            "utils.data_process",
        )
    return sorted(DATA_SOURCES.keys())


def get_supported_models() -> list[str]:
    """Return all registered model names from ``TRAINERS``.

    Lazily triggers discovery for ``model`` if the registry is still empty.
    """
    if not TRAINERS:
        from pathlib import Path

        discover_registrations(
            Path(__file__).resolve().parent.parent.parent / "model",
            "model",
        )
    return sorted(TRAINERS.keys())


def get_supported_stages() -> list[str]:
    """Return all registered efficiency stage names from ``EFFICIENCY_STAGES``.

    Lazily triggers discovery for ``utils.efficiency.stages`` if the registry is
    still empty (mirrors :func:`get_supported_models`).
    """
    if not EFFICIENCY_STAGES:
        from pathlib import Path

        discover_registrations(
            Path(__file__).resolve().parent.parent / "efficiency" / "stages",
            "utils.efficiency.stages",
        )
    return sorted(EFFICIENCY_STAGES.keys())


__all__ = [
    "ANALYZERS",
    "CASE_SELECTORS",
    "CASE_SINKS",
    "CASE_VISUALIZERS",
    "DATA_SOURCES",
    "EFFICIENCY_STAGES",
    "LLM_CLIENTS",
    "METRICS",
    "METRIC_LOGGERS",
    "MODEL_CONFIGS",
    # Registries
    "TRAINERS",
    "UniversalRegistry",
    "add_file_handler",
    # Discovery
    "discover_registrations",
    # Logger
    "get_logger",
    # Helpers
    "get_supported_datasets",
    "get_supported_models",
    "get_supported_stages",
    "load_env",
    "register_analyzer",
    "register_case_selector",
    "register_case_sink",
    "register_case_visualizer",
    "register_data_source",
    "register_efficiency_stage",
    "register_llm_client",
    "register_metric",
    "register_metric_logger",
    "register_model_config",
    # Decorators
    "register_trainer",
    "remove_file_handler",
    "reset_loggers",
    # Random
    "seed_everything",
    "set_log_level",
]
