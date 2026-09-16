"""Background resource sampling, descriptor-driven and stage-scoped.

``MetricSampler`` is a generic periodic background thread that probes a set of
resource metrics (CPU/RAM via psutil, GPU via NVML). ``StageScopedSampler``
wraps it to route samples per stage and aggregate them. ``RESOURCE_METRICS``
is the single descriptor table that drives probing, aggregation, and rendering
— adding a metric is one entry, not edits across four sites.
"""

import contextlib
import os
import statistics
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceSummary:
    """Single metric aggregation: mean/peak/min/p50 + sample count."""

    mean: float | None = None
    peak: float | None = None
    min: float | None = None
    p50: float | None = None
    n: int = 0


@dataclass
class ResourceStats:
    """Per-stage resource sampling aggregation (named fields, stable JSON schema)."""

    cpu_percent: ResourceSummary = field(default_factory=ResourceSummary)
    process_rss_mib: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_util_pct: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_power_w: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_mem_used_mib: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_temp_c: ResourceSummary = field(default_factory=ResourceSummary)
    gpu_sm_clock_mhz: ResourceSummary = field(default_factory=ResourceSummary)


@dataclass(frozen=True)
class ResourceMetric:
    """One sampled resource metric: metadata + a probe factory.

    ``make_probe(proc, nv_handle)`` builds the zero-arg probe callable once the
    psutil process / NVML handle are known at sampler start.
    """

    stats_field: str
    bucket_key: str
    label: str
    unit: str
    make_probe: Callable[[object, object], Callable[[], float | None]]


def _make_cpu_probe(proc: Any, _nv_handle: Any) -> Callable[[], float | None]:
    def probe() -> float | None:
        if proc is None:
            return None
        try:
            import psutil

            cpu = proc.cpu_percent(interval=None)
            for child in proc.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu += child.cpu_percent(interval=None)
            return cpu
        except Exception:
            return None

    return probe


def _make_rss_probe(proc: Any, _nv_handle: Any) -> Callable[[], float | None]:
    def probe() -> float | None:
        if proc is None:
            return None
        try:
            import psutil

            rss = proc.memory_info().rss / 1024**2
            for child in proc.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    rss += child.memory_info().rss / 1024**2
            return rss
        except Exception:
            return None

    return probe


def _make_gpu_probe(
    nv_call: Callable[[Any, Any], float],
) -> Callable[[object, object], Callable[[], float | None]]:
    """Build a probe factory that calls ``nv_call(pynvml, handle)`` each tick."""

    def make(_proc: Any, nv_handle: Any) -> Callable[[], float | None]:
        def probe() -> float | None:
            if nv_handle is None:
                return None
            try:
                import pynvml

                return float(nv_call(pynvml, nv_handle))
            except Exception:
                return None

        return probe

    return make


def _util(pynvml: Any, handle: Any) -> float:
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


def _mem(pynvml: Any, handle: Any) -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2


def _power(pynvml: Any, handle: Any) -> float:
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def _temp(pynvml: Any, handle: Any) -> int:
    return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)


def _sm_clock(pynvml: Any, handle: Any) -> int:
    return pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)


#: Single source of truth for resource metrics. Drives probing, aggregation,
#: and table rendering — each maps a ResourceStats field to its bucket key and
#: display label/unit.
RESOURCE_METRICS: list[ResourceMetric] = [
    ResourceMetric("cpu_percent", "cpu", "CPU%", "", _make_cpu_probe),
    ResourceMetric("process_rss_mib", "rss", "Process RSS", "MiB", _make_rss_probe),
    ResourceMetric("gpu_util_pct", "gpu_util", "GPU util", "%", _make_gpu_probe(_util)),
    ResourceMetric(
        "gpu_power_w", "gpu_power", "GPU power", "W", _make_gpu_probe(_power)
    ),
    ResourceMetric(
        "gpu_mem_used_mib", "gpu_mem", "GPU mem used", "MiB", _make_gpu_probe(_mem)
    ),
    ResourceMetric("gpu_temp_c", "gpu_temp", "GPU temp", "C", _make_gpu_probe(_temp)),
    ResourceMetric(
        "gpu_sm_clock_mhz",
        "gpu_sm_clock",
        "GPU SM clock",
        "MHz",
        _make_gpu_probe(_sm_clock),
    ),
]


class MetricSampler:
    """Generic periodic background thread probing a fixed set of resource metrics.

    Owns psutil/NVML lifecycle and probing; emits one sample dict
    (``{bucket_key: value}``) per tick to the installed ``sink``. Reusable for
    non-stage-scoped sampling.
    """

    def __init__(self, device: torch.device, interval_s: float = 0.05) -> None:
        """Configure the sampler device and tick interval (resources build at start)."""
        self.device = device
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink: Callable[[dict[str, float]], None] | None = None
        self._probes: list[tuple[str, Callable[[], float | None]]] = []
        self._nvml_inited = False

    def set_sink(self, sink: Callable[[dict[str, float]], None]) -> None:
        """Install the per-sample consumer (e.g. a stage-routing callback)."""
        self._sink = sink

    def start(self) -> None:
        """Init psutil/NVML, build probes, and start the sampling thread."""
        proc = None
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            # First cpu_percent call has no baseline (returns 0); prime it here.
            proc.cpu_percent(interval=None)
            for child in proc.children(recursive=True):
                child.cpu_percent(interval=None)
        except Exception as e:
            logger.warning(
                f"[Setup] psutil unavailable, CPU/RAM sampling disabled: {e}"
            )

        nv_handle = None
        if self.device.type == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                # Mark NVML initialized so stop() pairs the shutdown even if GetHandle fails.
                self._nvml_inited = True
                idx = self.device.index or 0
                nv_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            except Exception as e:
                logger.warning(
                    f"[Setup] pynvml unavailable, GPU sampling disabled: {e}"
                )

        self._probes = [
            (m.bucket_key, m.make_probe(proc, nv_handle)) for m in RESOURCE_METRICS
        ]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def _sample_once(self) -> None:
        if self._sink is None:
            return
        sample = {}
        for key, probe in self._probes:
            value = probe()
            if value is not None:
                sample[key] = value
        if sample:
            self._sink(sample)

    def stop(self) -> None:
        """Stop the sampling thread and release NVML."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml_inited:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass


class StageScopedSampler:
    """Route :class:`MetricSampler` samples per stage and aggregate on stop.

    ``begin_stage``/``end_stage`` direct each sample into the named stage's
    bucket; ``stop`` returns ``{stage: ResourceStats}`` aggregated from the
    descriptor table.
    """

    def __init__(self, device: torch.device, interval_s: float = 0.05) -> None:
        """Wrap a :class:`MetricSampler` for per-stage routing and aggregation."""
        self._metric = MetricSampler(device, interval_s)
        self._metric.set_sink(self._route)
        # Per-stage sample buckets keyed by metric bucket_key.
        self._buckets: dict[str, dict[str, list[float]]] = {}
        self._current: str | None = None

    def start(self) -> None:
        """Start the underlying metric sampler."""
        self._metric.start()

    def begin_stage(self, name: str) -> None:
        """Route subsequent samples to the named stage's bucket."""
        if name not in self._buckets:
            self._buckets[name] = {m.bucket_key: [] for m in RESOURCE_METRICS}
        self._current = name

    def end_stage(self) -> None:
        """Stop attributing samples until the next stage begins."""
        self._current = None

    def _route(self, sample: dict[str, float]) -> None:
        current = self._current
        if current is None:
            return
        bucket = self._buckets.get(current)
        if bucket is None:
            return
        for key, value in sample.items():
            bucket.setdefault(key, []).append(value)

    def stop(self) -> dict[str, ResourceStats]:
        """Stop sampling and return per-stage aggregated ResourceStats."""
        self._metric.stop()
        return {
            name: ResourceStats(
                **{
                    m.stats_field: _summarize(bucket.get(m.bucket_key, []))
                    for m in RESOURCE_METRICS
                }
            )
            for name, bucket in self._buckets.items()
        }


#: Backward-compatible alias; the stage-scoped sampler is the resource sampler.
ResourceSampler = StageScopedSampler


def _summarize(xs: list[float]) -> ResourceSummary:
    """Aggregate samples into mean/peak/min/p50."""
    if not xs:
        return ResourceSummary()
    return ResourceSummary(
        mean=sum(xs) / len(xs),
        peak=max(xs),
        min=min(xs),
        p50=statistics.median(xs),
        n=len(xs),
    )
