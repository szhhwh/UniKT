"""Resource monitoring service with NVML-based GPU status and system metrics.

A background sampler thread queries NVIDIA GPU devices via pynvml and system
resources via psutil every few seconds, keeping a ring buffer of samples so
requests read cached snapshots and clients get server-side trend history.
"""

import contextlib
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

import psutil
import pynvml
from schemas import (
    GpuHistorySeries,
    GpuInfo,
    GpuStatusResponse,
    ResourceHistoryResponse,
    ResourceSnapshot,
    SystemStatusResponse,
)

logger = logging.getLogger(__name__)

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _fmt_time(ts_seconds: float) -> str:
    """Format an epoch timestamp the same way the previous live queries did."""
    return time.strftime(_TIMESTAMP_FORMAT, time.localtime(ts_seconds))


@dataclass
class _Sample:
    """One full resource sample; gpus is empty when NVML misses."""

    ts: int  # epoch milliseconds
    gpus: list[GpuInfo]
    cpu: float
    cpu_cores: tuple[float, ...]
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    swap_percent: float
    swap_used_gb: float
    swap_total_gb: float
    net_up_bps: float
    net_down_bps: float
    disk_read_bps: float
    disk_write_bps: float
    load: tuple[float, float, float]


class GpuMonitor:
    """Monitors GPU and system status via a background sampling thread.

    Args:
        sample_seconds: Interval between samples (default 2.0).
        history_seconds: How far back to keep samples (default 900.0).
    """

    def __init__(self, sample_seconds: float = 2.0, history_seconds: float = 900.0):
        """Initialize NVML, take a first synchronous sample, start the sampler.

        Args:
            sample_seconds: Interval between samples in seconds.
            history_seconds: History retention window in seconds.
        """
        self._sample_seconds = sample_seconds
        self._samples: deque[_Sample] = deque(
            maxlen=int(history_seconds / sample_seconds) + 1
        )
        self._latest: _Sample | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._prev_net: tuple[float, float, float] | None = None
        self._prev_disk: tuple[float, float, float] | None = None
        self._nvml_initialized = False
        self._device_count = 0
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            pass
        except Exception:
            logger.exception("Unexpected error initialising NVML")
        # First sample synchronously so get_status has data immediately; its
        # CPU/net/disk rates read 0.0 because the diff windows have no past.
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run, name="resource-sampler", daemon=True
        )
        self._thread.start()

    @property
    def device_count(self) -> int:
        """Stable GPU count captured at init.

        Used for scheduling-lane sizing so a transient NVML miss in a sample
        (which stores an empty gpu list) cannot collapse all GPU lanes to the
        CPU lane mid-dispatch.
        """
        return self._device_count

    def _run(self) -> None:
        """Sample forever on a fixed interval until shutdown wakes the thread."""
        while not self._stopping:
            self._wake.wait(timeout=self._sample_seconds)
            if self._stopping:
                return
            try:
                self._sample_once()
            except Exception:
                logger.exception("Resource sampler error")

    def _sample_once(self) -> None:
        """Take one sample of every resource and append it to the ring buffer."""
        now = time.time()
        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        net_up = net_down = disk_read = disk_write = 0.0
        if net is not None:
            if self._prev_net is not None:
                dt = max(now - self._prev_net[0], 1e-6)
                net_up = max(net.bytes_sent - self._prev_net[1], 0) / dt
                net_down = max(net.bytes_recv - self._prev_net[2], 0) / dt
            self._prev_net = (now, net.bytes_sent, net.bytes_recv)
        if disk is not None:
            if self._prev_disk is not None:
                dt = max(now - self._prev_disk[0], 1e-6)
                disk_read = max(disk.read_bytes - self._prev_disk[1], 0) / dt
                disk_write = max(disk.write_bytes - self._prev_disk[2], 0) / dt
            self._prev_disk = (now, disk.read_bytes, disk.write_bytes)
        # Non-blocking mode: values cover the window since the previous call,
        # so the first sample after startup reports zeros by design.
        cores = tuple(psutil.cpu_percent(percpu=True) or ())
        cpu = sum(cores) / len(cores) if cores else 0.0
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        load1, load5, load15 = os.getloadavg()
        sample = _Sample(
            ts=int(now * 1000),
            gpus=self._query_nvml(),
            cpu=cpu,
            cpu_cores=cores,
            memory_percent=mem.percent,
            memory_used_gb=round(mem.used / 1024**3, 1),
            memory_total_gb=round(mem.total / 1024**3, 1),
            swap_percent=swap.percent,
            swap_used_gb=round(swap.used / 1024**3, 1),
            swap_total_gb=round(swap.total / 1024**3, 1),
            net_up_bps=round(net_up, 1),
            net_down_bps=round(net_down, 1),
            disk_read_bps=round(disk_read, 1),
            disk_write_bps=round(disk_write, 1),
            load=(round(load1, 2), round(load5, 2), round(load15, 2)),
        )
        with self._lock:
            self._samples.append(sample)
            self._latest = sample

    def get_status(self) -> GpuStatusResponse:
        """Return the latest sampled GPU status without touching NVML."""
        with self._lock:
            latest = self._latest
        if latest is None:
            return GpuStatusResponse(gpus=[], updated_at=_fmt_time(time.time()))
        return GpuStatusResponse(
            gpus=[g.model_copy() for g in latest.gpus],
            updated_at=_fmt_time(latest.ts / 1000),
        )

    def _query_nvml(self) -> list[GpuInfo]:
        """Query all NVIDIA GPUs with NVML.

        Returns:
            A list of GpuInfo objects, or an empty list if NVML is unavailable.
        """
        if not self._nvml_initialized:
            return []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    power = 0.0
                gpus.append(
                    GpuInfo(
                        index=i,
                        name=name,
                        utilization_percent=float(util.gpu),
                        memory_used_mb=round(mem.used / (1024 * 1024), 1),
                        memory_total_mb=round(mem.total / (1024 * 1024), 1),
                        temperature_c=float(temp),
                        power_usage_w=round(power, 1),
                        processes=[],
                    )
                )
            return gpus
        except pynvml.NVMLError:
            return []
        except Exception:
            logger.exception("Unexpected error querying NVML")
            return []

    def get_system_status(self) -> SystemStatusResponse:
        """Return the latest sampled system status with a multi-GPU mean."""
        with self._lock:
            latest = self._latest
        if latest is None:
            return SystemStatusResponse(
                cpu_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                memory_percent=0.0,
                gpu_utilization=0.0,
                gpu_memory_percent=0.0,
                load_1m=0.0,
                load_5m=0.0,
                load_15m=0.0,
                updated_at=_fmt_time(time.time()),
            )
        utils = [g.utilization_percent for g in latest.gpus]
        mems = [
            g.memory_used_mb / g.memory_total_mb * 100
            for g in latest.gpus
            if g.memory_total_mb > 0
        ]
        return SystemStatusResponse(
            cpu_percent=round(latest.cpu, 1),
            memory_used_gb=latest.memory_used_gb,
            memory_total_gb=latest.memory_total_gb,
            memory_percent=latest.memory_percent,
            gpu_utilization=round(sum(utils) / len(utils), 1) if utils else 0.0,
            gpu_memory_percent=round(sum(mems) / len(mems), 1) if mems else 0.0,
            load_1m=latest.load[0],
            load_5m=latest.load[1],
            load_15m=latest.load[2],
            updated_at=_fmt_time(latest.ts / 1000),
        )

    def get_history(self, since_ms: int | None = None) -> ResourceHistoryResponse:
        """Return sampled metric history, optionally only samples after since_ms.

        Args:
            since_ms: Epoch-millisecond cursor; samples with ts <= since_ms are
                skipped. The snapshot is always the global latest regardless.

        Returns:
            A ResourceHistoryResponse with column-oriented series.
        """
        with self._lock:
            samples = list(self._samples)
            latest = self._latest
            gpu_names: list[str] = []
            for s in reversed(samples):
                if s.gpus:
                    gpu_names = [g.name for g in s.gpus]
                    break
        if since_ms is not None:
            samples = [s for s in samples if s.ts > since_ms]

        def col(attr: str) -> list[float]:
            return [getattr(s, attr) for s in samples]

        gpus = [
            GpuHistorySeries(
                index=i,
                name=gpu_names[i] if i < len(gpu_names) else f"GPU {i}",
                utilization_percent=[
                    s.gpus[i].utilization_percent if i < len(s.gpus) else None
                    for s in samples
                ],
                memory_percent=[
                    round(s.gpus[i].memory_used_mb / s.gpus[i].memory_total_mb * 100, 1)
                    if i < len(s.gpus) and s.gpus[i].memory_total_mb > 0
                    else None
                    for s in samples
                ],
            )
            for i in range(self._device_count)
        ]
        if latest is not None:
            snapshot = ResourceSnapshot(
                cpu_percent=round(latest.cpu, 1),
                cpu_cores=[round(c) for c in latest.cpu_cores],
                load_1m=latest.load[0],
                load_5m=latest.load[1],
                load_15m=latest.load[2],
                memory_used_gb=latest.memory_used_gb,
                memory_total_gb=latest.memory_total_gb,
                memory_percent=latest.memory_percent,
                swap_used_gb=latest.swap_used_gb,
                swap_total_gb=latest.swap_total_gb,
                swap_percent=latest.swap_percent,
            )
        else:
            snapshot = ResourceSnapshot(
                cpu_percent=0.0,
                cpu_cores=[],
                load_1m=0.0,
                load_5m=0.0,
                load_15m=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                memory_percent=0.0,
                swap_used_gb=0.0,
                swap_total_gb=0.0,
                swap_percent=0.0,
            )
        return ResourceHistoryResponse(
            timestamps=[s.ts for s in samples],
            cpu_percent=col("cpu"),
            memory_percent=col("memory_percent"),
            swap_percent=col("swap_percent"),
            net_up_bps=col("net_up_bps"),
            net_down_bps=col("net_down_bps"),
            disk_read_bps=col("disk_read_bps"),
            disk_write_bps=col("disk_write_bps"),
            gpus=gpus,
            snapshot=snapshot,
            interval_seconds=self._sample_seconds,
        )

    def shutdown(self) -> None:
        """Stop the sampler thread and release NVML resources.

        Safe to call multiple times; suppresses errors.
        """
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout=5)
        if self._nvml_initialized:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
