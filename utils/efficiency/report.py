"""Efficiency report: stage-agnostic container + JSON/Rich console output.

The report holds each stage's result under its registry name in ``results``;
console rendering looks the stage up in ``EFFICIENCY_STAGES`` and asks it to
format its own table. Adding a stage never touches this file — its result and
table appear automatically.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from utils.core import EFFICIENCY_STAGES, get_logger

from .environment import (
    RESOURCE_METRICS,
    EnvironmentInfo,
    ResourceStats,
    ResourceSummary,
)
from .stages.base import TITLE_STYLE, EfficiencyStage

logger = get_logger(__name__)

# Console cap for a failed stage's error message (JSON keeps the full text).
_CONSOLE_ERROR_MAX_CHARS = 200


@dataclass
class EfficiencyReport:
    """Stage-agnostic efficiency report."""

    model_name: str
    dataset_name: str
    timestamp: str
    batch_size: int
    seq_len: int | None
    modes: list[str]
    config: dict[str, Any]
    determinism: dict[str, Any]
    environment: EnvironmentInfo
    resource: dict[str, ResourceStats] | None
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the report (including nested stage results) to a dict."""
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write the report as JSON, creating parent dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"[Report] saved to {path}")

    def print_console(self) -> None:
        """Print the report: one table per stage result (failed stages render their error), then resource + environment."""
        console = Console()
        console.print()
        console.print(
            f"[bold cyan]Efficiency Report[/]  "
            f"[white]{self.model_name}[/] on [white]{self.dataset_name}[/]  "
            f"({self.timestamp})"
        )
        console.print(
            f"  batch_size={self.batch_size}  seq_len={self.seq_len}  "
            f"modes={','.join(self.modes)}"
        )
        console.print()

        for name in self.modes:
            if name in self.errors:
                console.print(_stage_error_table(name, self.errors[name]))
                console.print()
                continue
            result = self.results.get(name)
            if result is None:
                continue
            stage_cls = cast(type[EfficiencyStage], EFFICIENCY_STAGES.get(name))
            table = stage_cls.format_table(result)
            if table is not None:
                console.print(table)
                console.print()

        for stage_name in self.modes:
            stats = (self.resource or {}).get(stage_name)
            if stats is None:
                continue
            console.print(
                _resource_table(stats, title=f"Resource Usage — {stage_name}")
            )
            console.print()
        console.print(_environment_table(self.environment))
        console.print()


def _stage_error_table(name: str, message: str) -> Table:
    """Render a failed stage's error (red, single line, capped for the console).

    The JSON report keeps the full multi-line message; only the console view
    is collapsed/truncated.
    """
    one_line = " ".join(message.split())
    if len(one_line) > _CONSOLE_ERROR_MAX_CHARS:
        one_line = one_line[: _CONSOLE_ERROR_MAX_CHARS - 1] + "…"
    table = Table(
        title=f"Stage failed — {name}",
        title_style="bold red",
        show_header=False,
        box=None,
    )
    table.add_column("Error", style="red")
    table.add_row(one_line)
    return table


def _resource_table(stats: ResourceStats, title: str = "Resource Usage") -> Table:
    table = Table(title=title, title_style=TITLE_STYLE, show_header=False, box=None)
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Mean", style="white")
    table.add_column("Peak", style="white")
    for metric in RESOURCE_METRICS:
        table.add_row(
            metric.label, *_rs(getattr(stats, metric.stats_field), metric.unit)
        )
    if all(getattr(stats, m.stats_field).n == 0 for m in RESOURCE_METRICS):
        table.add_row("(no samples)", "", "")
    return table


def _rs(summary: ResourceSummary, unit: str = "") -> tuple[str, str]:
    if summary.n == 0:
        return ("—", "—")
    mean = f"{summary.mean:.1f} {unit}".strip()
    peak = f"{summary.peak:.1f} {unit}".strip()
    return (mean, peak)


def _environment_table(env: EnvironmentInfo) -> Table:
    table = Table(
        title="Environment", title_style=TITLE_STYLE, show_header=False, box=None
    )
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Device", env.device_type)
    if env.gpu_name:
        table.add_row("GPU", f"{env.gpu_name} ({env.gpu_capability})")
    if env.gpu_total_memory_mib is not None:
        table.add_row("GPU memory", f"{env.gpu_total_memory_mib:,.0f} MiB")
    if env.gpu_max_sm_clock_mhz:
        table.add_row("GPU max SM clock", f"{env.gpu_max_sm_clock_mhz} MHz")
    table.add_row(
        "CPU",
        f"{env.cpu_model or 'unknown'} ({env.cpu_physical_cores or '?'}c/{env.cpu_logical_cores or '?'}t)",
    )
    if env.total_ram_gib:
        table.add_row("RAM", f"{env.total_ram_gib:.1f} GiB")
    table.add_row("PyTorch", env.torch_version)
    if env.cuda_version:
        table.add_row("CUDA", env.cuda_version)
    if env.cudnn_version:
        table.add_row(
            "cuDNN",
            f"{env.cudnn_version} (benchmark={env.cudnn_benchmark}, det={env.cudnn_deterministic})",
        )
    table.add_row("Deterministic algos", str(env.deterministic_algorithms))
    if env.cuda_available:
        table.add_row("TF32 (matmul)", str(env.cuda_matmul_allow_tf32))
        table.add_row("TF32 (cuDNN)", str(env.cudnn_allow_tf32))
    table.add_row("Matmul precision", env.float32_matmul_precision or "default")
    if env.model_dtype:
        table.add_row("Model dtype", env.model_dtype)
    table.add_row("Python", env.python_version)
    table.add_row("Platform", env.platform)
    return table
