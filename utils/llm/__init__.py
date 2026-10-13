"""Unified LLM inference layer: capability clients + persistent cache.

Model-agnostic service for trainers (offline enrichment in
``build_components``), data processing, and analysis entry points; see
:mod:`utils.llm.base` for the client contract. Backends register via
``@register_llm_client`` and are discovered statically like trainers.

``utils.llm.usage`` (trainer observability) and the litellm transport in
``utils.llm.litellm_client`` are imported explicitly by consumers: the
former pulls the training callback stack (torch), the latter litellm
itself — importing this package pulls neither.
"""

from pathlib import Path

from utils.core import discover_registrations

from .base import (
    ChoiceRequest,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    TokenLogprob,
    create_llm_client,
)
from .cache import SQLiteResponseCache, cache_key
from .mock_client import MockLLMClient

discover_registrations(Path(__file__).parent, "utils.llm")

__all__ = [
    "ChoiceRequest",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "MockLLMClient",
    "SQLiteResponseCache",
    "TokenLogprob",
    "cache_key",
    "create_llm_client",
]
