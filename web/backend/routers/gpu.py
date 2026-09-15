"""GPU router — real-time GPU and system status endpoints.

Returns cached GPU status (utilization, memory, temperature) enriched with the
experiment tasks occupying each GPU, plus system-level status (CPU, memory,
load averages).
"""

from typing import Any

from database import SessionLocal
from dependencies import get_gpu_monitor
from fastapi import APIRouter, Depends
from models import Task
from schemas import GpuStatusResponse, SystemStatusResponse
from services.gpu_monitor import GpuMonitor

router = APIRouter(prefix="/api/gpu", tags=["gpu"])


@router.get("/status", response_model=GpuStatusResponse)
def gpu_status(monitor: GpuMonitor = Depends(get_gpu_monitor)) -> Any:
    """Return current GPU status with the tasks occupying each GPU.

    Occupancy is derived from the task dispatch records rather than NVML
    process listings, so each process maps back to a known experiment task.

    Args:
        monitor: Injected GpuMonitor singleton.

    Returns:
        A GpuStatusResponse with per-GPU metrics and occupying tasks.
    """
    status = monitor.get_status()
    occupancy: dict[int, list[dict]] = {}
    with SessionLocal() as session:
        rows = (
            session.query(Task.id, Task.name, Task.status, Task.pid, Task.gpu_assigned)
            .filter(Task.status.in_(["running", "stopping", "interrupted"]))
            .filter(Task.gpu_assigned.is_not(None))
            .all()
        )
    for tid, name, tstatus, pid, gpu in rows:
        occupancy.setdefault(gpu, []).append(
            {"id": tid, "name": name, "status": tstatus, "pid": pid}
        )
    gpus = [
        gpu.model_copy(update={"processes": occupancy.get(gpu.index, [])})
        for gpu in status.gpus
    ]
    return GpuStatusResponse(gpus=gpus, updated_at=status.updated_at)


@router.get("/system", response_model=SystemStatusResponse)
def system_status(monitor: GpuMonitor = Depends(get_gpu_monitor)) -> Any:
    """Return current system status (CPU, memory, load, GPU aggregate).

    Args:
        monitor: Injected GpuMonitor singleton.

    Returns:
        A SystemStatusResponse with CPU, memory, and GPU metrics.
    """
    return monitor.get_system_status()
