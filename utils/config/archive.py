"""RunConfig yaml archive: lossless save/load for reproducibility.

The config tree serializes to ``run_config.yaml``; runtime-derived metadata
(param count, optimizer, device info) that does not belong in config goes to a
sidecar ``run_metadata.yaml``.

Load re-constructs the typed :class:`RunConfig` tree against the concrete
model's schema, so a stale archive (unknown/removed field) raises at load time.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .run_config import RunConfig, build_run_config_schema, config_to_dict


def save_run_config_archive(
    rc: Any,
    log_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    """Write a :class:`RunConfig` to ``<log_dir>/run_config.yaml``.

    Args:
        rc: RunConfig instance.
        log_dir: Output directory (created if missing).
        metadata: Optional runtime-derived metadata written to a sidecar
            ``run_metadata.yaml`` (e.g. total_params, optimizer, device info).

    Returns:
        ``(config_path, metadata_path_or_none)``.
    """
    out = Path(log_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "run_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_to_dict(rc), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    metadata_path: Path | None = None
    if metadata:
        metadata_path = out / "run_metadata.yaml"
        metadata_path.write_text(
            yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return config_path, metadata_path


def load_run_config_archive(yaml_path: str | Path) -> RunConfig:
    """Load a ``run_config.yaml`` and reconstruct the typed :class:`RunConfig`.

    The model name is read from ``experiment.model_name``; constructing each node
    dataclass against the current schema validates that archived fields still
    exist (unknown/removed fields raise ``TypeError``).
    """
    yaml_path = Path(yaml_path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    model_name = data.get("experiment", {}).get("model_name", "")
    schema_nodes = build_run_config_schema(model_name)

    extra_top = sorted(set(data) - set(schema_nodes))
    if extra_top:
        raise TypeError(
            f"Archived run_config.yaml has top-level node(s) absent from the current "
            f"schema: {extra_top}"
        )
    node_instances: dict[str, Any] = {}
    for node, cls in schema_nodes.items():
        node_instances[node] = _build_node_from_dict(cls, data.get(node, {}))
    return RunConfig(**node_instances)


def load_run_metadata(log_dir: str | Path) -> dict[str, Any]:
    """Read the sidecar ``run_metadata.yaml`` if present, else return ``{}``."""
    path = Path(log_dir) / "run_metadata.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def update_run_metadata(log_dir: str | Path, updates: dict[str, Any]) -> None:
    """Merge ``updates`` into the sidecar ``run_metadata.yaml`` (top-level keys).

    The sidecar is written at archive time with run-start facts; values only
    known at the end of a run (e.g. LLM token usage) extend it here. Keys not
    present in ``updates`` are preserved.

    Args:
        log_dir: Run directory holding ``run_metadata.yaml``.
        updates: Top-level keys to set or replace.
    """
    merged = load_run_metadata(log_dir)
    merged.update(updates)
    path = Path(log_dir) / "run_metadata.yaml"
    path.write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _build_node_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Construct a config dataclass from a dict, raising on unknown archived keys."""
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise TypeError(
            f"Archived {cls.__name__} has field(s) absent from the current schema "
            f"(likely renamed/removed): {unknown}"
        )
    return cls(**data)


__all__ = [
    "load_run_config_archive",
    "load_run_metadata",
    "save_run_config_archive",
    "update_run_metadata",
]
