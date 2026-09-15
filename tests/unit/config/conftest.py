"""Shared fixtures for config-area tests: yaml config writer factory."""

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def write_model_yaml(tmp_path: Path) -> Callable[..., Path]:
    """Factory: write a --config yaml naming a model (plus optional overrides)."""

    def _write(
        model_name: str = "TinyTestModel",
        overrides: dict[str, object] | None = None,
        filename: str = "config.yaml",
    ) -> Path:
        import yaml

        data: dict[str, dict[str, object]] = {"experiment": {"model_name": model_name}}
        for dotted, value in (overrides or {}).items():
            node, _, field = dotted.partition(".")
            data.setdefault(node, {})[field] = value
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write
