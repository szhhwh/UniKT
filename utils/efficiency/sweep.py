"""Axis sweep: run the efficiency benchmark across a set of config mutations.

A sweep is a list of :class:`SweepPoint` ``(label, mutate)`` entries. Each point
runs on a fresh deepcopy of the run config with its ``mutate`` applied, rebuilds
a trainer under its own sub-dir (clean CUDA allocator, isolated peak memory),
and prints its full efficiency report. :func:`batch_size_sweep` is the built-in
factory; ``seq_len``/precision/fold sweeps are the same shape.

A lightweight index lists the runs (and any points that failed, with their
error); per-point metrics stay in each point's own report (no cross-point
aggregation).
"""

from __future__ import annotations

import copy
import gc
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from rich.console import Console

from utils.config import CompileConfig, RunConfig, config_to_dict
from utils.core import add_file_handler, get_logger
from utils.experiment_manager import ExperimentManager, ExperimentType

if TYPE_CHECKING:
    from utils.data_process import DataSource

from .report import EfficiencyReport
from .session import EfficiencySession, build_target

logger = get_logger(__name__)

CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]


@dataclass
class SweepPoint:
    """One sweep axis point: a label and a config mutator.

    ``mutate(rc, eff_cfg)`` applies absolute values to the (already deepcopied)
    run/efficiency config, so points are independent and need no restore.
    """

    label: str
    mutate: Callable[..., None]


@dataclass
class SweepRun:
    """Location of one point's efficiency report within the sweep dir."""

    label: str
    dir: str


@dataclass
class SweepFailure:
    """One sweep point that raised: its label and a one-line error summary."""

    label: str
    error: str


@dataclass
class SweepReport:
    """Lightweight sweep index: which points ran and which failed, with locations.

    Holds no per-point metrics — each point's full EfficiencyReport stays in its
    own sub-directory; this index only points to them.
    """

    model_name: str
    dataset_name: str
    timestamp: str
    labels: list[str]
    modes: list[str]
    config: dict
    sweep_dir: str
    runs: list[SweepRun]
    failures: list[SweepFailure] = field(default_factory=list)

    def write_json(self, path: str | Path) -> None:
        """Write the sweep index as JSON, creating parent dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "modes": self.modes,
            "config": self.config,
            "sweep_dir": self.sweep_dir,
            "runs": [{"label": r.label, "dir": r.dir} for r in self.runs],
            "failures": [{"label": f.label, "error": f.error} for f in self.failures],
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[Sweep] index saved to {path}")


def batch_size_sweep(sizes: list[int]) -> list[SweepPoint]:
    """Build sweep points that vary ``rc.model.batch_size`` across ``sizes``."""
    return [
        SweepPoint(
            f"bs{size}",
            lambda rc, _eff_cfg, s=size: setattr(rc.model, "batch_size", s),
        )
        for size in sizes
    ]


# ``off`` is an efficiency-framework sentinel (compile disabled, the baseline);
# the rest are valid ``torch.compile`` modes (see CompileConfig.compile_mode).
_VALID_COMPILE_MODES = [
    "off",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]


def compile_sweep(modes: list[str]) -> list[SweepPoint]:
    """Build sweep points that vary ``rc.compile`` across the given states.

    ``off`` disables compile (baseline); any other entry enables compile with
    that ``compile_mode``. The remaining CompileConfig knobs (fullgraph /
    dynamic / backend) pass through, so ``--compile.*`` still composes.
    """
    return [
        SweepPoint(
            f"cmp_{mode}",
            lambda rc, _eff_cfg, m=mode: _apply_compile_state(rc.compile, m),
        )
        for mode in modes
    ]


def _apply_compile_state(cc: CompileConfig, mode: str) -> None:
    """Set ``cc`` to the swept compile state: off, or on with ``mode``."""
    if mode == "off":
        cc.compile = False
    else:
        cc.compile = True
        # mode validity is enforced upstream (_parse_compile_modes allow-list).
        cc.compile_mode = cast(CompileMode, mode)


def cartesian_sweep(
    axis_a: list[SweepPoint], axis_b: list[SweepPoint]
) -> list[SweepPoint]:
    """Combine two sweep axes into their Cartesian product.

    Each product point applies both mutators (order-independent when they touch
    disjoint config fields, as batch_size and compile do) and is labeled
    ``<a>_<b>`` (e.g. ``bs32_cmp_off``).
    """
    product: list[SweepPoint] = []
    for a in axis_a:
        for b in axis_b:

            def mutate(
                rc: RunConfig,
                eff_cfg: Any,
                ma: Callable[..., None] = a.mutate,
                mb: Callable[..., None] = b.mutate,
            ) -> None:
                ma(rc, eff_cfg)
                mb(rc, eff_cfg)

            product.append(SweepPoint(f"{a.label}_{b.label}", mutate))
    return product


class EfficiencySweep:
    """Run the efficiency benchmark across a set of :class:`SweepPoint` entries.

    Defaults to :func:`batch_size_sweep` parsed from
    ``eff_cfg.general.batch_sizes``; pass ``points`` for any other axis.
    """

    def __init__(
        self,
        rc: RunConfig,
        eff_cfg: Any,
        data_src: DataSource,
        weights_path: str | None = None,
        points: list[SweepPoint] | None = None,
    ) -> None:
        """Bind run config, data source, optional weights, and sweep points."""
        self.rc = rc
        self.cfg = eff_cfg
        self.data_src = data_src
        self.weights_path = weights_path
        self.points = self._resolve_points(points, eff_cfg)
        tags = [f"fold{rc.data.fold}"] if rc.data.fold is not None else []
        tags.append("sweep")
        self.parent_exp = ExperimentManager(
            exp_type=ExperimentType.EFFICIENCY,
            model_name=rc.experiment.model_name,
            dataset_name=rc.data.dataset,
            base_dir="runs",
            tags=tags,
        )
        self.sweep_dir = self.parent_exp.get_log_dir()

    @staticmethod
    def _resolve_points(
        points: list[SweepPoint] | None, eff_cfg: Any
    ) -> list[SweepPoint]:
        """Resolve sweep points from the configured axes.

        Explicit points win; otherwise ``batch_sizes`` and/or ``compile_modes``
        (both set → their Cartesian product; one set → that single axis).
        """
        if points is not None:
            return points
        compile_modes = eff_cfg.general.compile_modes
        batch_sizes = eff_cfg.general.batch_sizes
        if compile_modes and batch_sizes:
            return cartesian_sweep(
                batch_size_sweep(_parse_batch_sizes(batch_sizes)),
                compile_sweep(_parse_compile_modes(compile_modes)),
            )
        if compile_modes:
            return compile_sweep(_parse_compile_modes(compile_modes))
        return batch_size_sweep(_parse_batch_sizes(batch_sizes))

    def run(self) -> SweepReport:
        """Run every point (printing each report) and index the runs."""
        add_file_handler(Path(self.sweep_dir) / "run.log")
        if self.cfg.general.output_dir:
            logger.warning(
                "[Sweep] --efficiency.general.output_dir ignored in sweep mode; "
                "each point writes to <sweep_dir>/<label>/."
            )
        logger.info(
            f"[Sweep] sweep_dir={self.sweep_dir} points={[p.label for p in self.points]}"
        )

        runs: list[SweepRun] = []
        failures: list[SweepFailure] = []
        modes: list[str] = []
        for point in self.points:
            logger.info(f"[Sweep] === {point.label} ===")
            try:
                report, child_dir = self._run_point(point)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                failures.append(SweepFailure(point.label, error))
                logger.error(
                    f"[Sweep] {point.label} failed, skipping: {e}", exc_info=True
                )
                continue
            finally:
                self._cleanup_cuda()
            runs.append(SweepRun(point.label, child_dir))
            if not modes:
                modes = report.modes

        if not runs:
            raise SystemExit("[Sweep] no sweep point completed successfully.")
        sweep_report = SweepReport(
            model_name=self.rc.experiment.model_name,
            dataset_name=self.rc.data.dataset,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            labels=[r.label for r in runs],
            modes=modes,
            config=config_to_dict(self.cfg),
            sweep_dir=self.sweep_dir,
            runs=runs,
            failures=failures,
        )
        sweep_report.write_json(Path(self.sweep_dir) / "sweep_index.json")
        self._print_summary(sweep_report)
        return sweep_report

    def _run_point(self, point: SweepPoint) -> tuple[EfficiencyReport, str]:
        """Apply ``point`` to a fresh rc copy, rebuild, run + print its report."""
        rc = copy.deepcopy(self.rc)
        point.mutate(rc, self.cfg)
        child_exp = self.parent_exp.create_sub_experiment(point.label)
        # Point the shared file sink at this point's run.log so each point dir
        # is self-contained; the sweep-level sink is restored in finally.
        add_file_handler(Path(child_exp.get_log_dir()) / "run.log")
        try:
            target = build_target(rc, self.data_src, child_exp, self.weights_path)
            report = EfficiencySession(
                target=target,
                rc=rc,
                eff_cfg=self.cfg,
                output_dir=child_exp.get_log_dir(),
            ).run()
            report.print_console()
            return report, child_exp.get_log_dir()
        finally:
            add_file_handler(Path(self.sweep_dir) / "run.log")

    def _print_summary(self, sweep_report: SweepReport) -> None:
        """Print a short index of which points ran and where their reports live."""
        console = Console()
        console.print()
        console.print(
            f"[bold cyan]Efficiency Sweep[/] done  "
            f"[white]{sweep_report.model_name}[/] on [white]{sweep_report.dataset_name}[/]  "
            f"({sweep_report.timestamp})"
        )
        console.print(
            f"  points={','.join(sweep_report.labels)}  "
            f"modes={','.join(sweep_report.modes)}"
        )
        for run in sweep_report.runs:
            console.print(f"  {run.label} -> {run.dir}")
        for failure in sweep_report.failures:
            console.print(f"  [red]{failure.label} -> FAILED:[/] {failure.error}")
        console.print()

    def _cleanup_cuda(self) -> None:
        """Release the prior trainer's tensors and reset CUDA peak stats.

        ``reset_peak_memory_stats`` raises on CPU, so the CUDA calls stay guarded;
        ``gc.collect`` reclaims optimizer<->model cycles regardless of device.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def _parse_batch_sizes(s: str) -> list[int]:
    """Parse comma-separated batch sizes: validate int > 0, de-duplicate preserving order."""
    raw = [t.strip() for t in s.split(",") if t.strip()]
    if not raw:
        raise SystemExit("[Sweep] --efficiency.general.batch_sizes is empty.")
    sizes: list[int] = []
    for t in raw:
        try:
            v = int(t)
        except ValueError:
            raise SystemExit(f"[Sweep] invalid batch_size '{t}' (must be int).")
        if v <= 0:
            raise SystemExit(f"[Sweep] batch_size must be > 0, got {v}.")
        sizes.append(v)
    return list(dict.fromkeys(sizes))


def _parse_compile_modes(s: str) -> list[str]:
    """Parse comma-separated compile states: validate against the allow-list, de-duplicate preserving order."""
    raw = [t.strip() for t in s.split(",") if t.strip()]
    if not raw:
        raise SystemExit("[Sweep] --efficiency.general.compile_modes is empty.")
    modes: list[str] = []
    for t in raw:
        if t not in _VALID_COMPILE_MODES:
            raise SystemExit(
                f"[Sweep] invalid compile mode '{t}'. Valid: {_VALID_COMPILE_MODES}"
            )
        modes.append(t)
    return list(dict.fromkeys(modes))


__all__ = [
    "EfficiencySweep",
    "SweepPoint",
    "SweepReport",
    "batch_size_sweep",
    "cartesian_sweep",
    "compile_sweep",
]
