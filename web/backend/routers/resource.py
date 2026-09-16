"""Resource router — server-side metric history for the resource monitor page."""

from typing import Any

from dependencies import get_gpu_monitor
from fastapi import APIRouter, Depends
from schemas import ResourceHistoryResponse
from services.gpu_monitor import GpuMonitor

router = APIRouter(prefix="/api/resource", tags=["resource"])


@router.get("/history", response_model=ResourceHistoryResponse)
def resource_history(
    since: int | None = None, monitor: GpuMonitor = Depends(get_gpu_monitor)
) -> Any:
    """Return sampled resource history, optionally after an epoch-ms cursor.

    Args:
        since: Epoch-millisecond cursor; only samples newer than this are
            returned, enabling cheap incremental polling.
        monitor: Injected GpuMonitor singleton.

    Returns:
        A ResourceHistoryResponse with column-oriented metric series.
    """
    return monitor.get_history(since_ms=since)
