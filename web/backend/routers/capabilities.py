"""Capabilities router — system GPU detection endpoint.

Provides a single GET endpoint that detects available NVIDIA GPUs via
nvidia-smi and returns the count and model names, with caching.
"""

import subprocess
import threading
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/system", tags=["system"])


class CapabilitiesResponse(BaseModel):
    """Response model for system capabilities.

    Attributes:
        has_gpu: Whether at least one NVIDIA GPU was detected.
        gpu_count: Number of detected GPUs.
        gpu_names: List of GPU model name strings.
    """

    has_gpu: bool
    gpu_count: int
    gpu_names: list[str]


_cached_capabilities: CapabilitiesResponse | None = None
_capabilities_lock = threading.Lock()


def _detect_gpu() -> CapabilitiesResponse:
    """Detect NVIDIA GPUs by running nvidia-smi, with caching.

    Returns:
        A CapabilitiesResponse with GPU detection results. Results are cached
        after the first successful detection.
    """
    global _cached_capabilities
    if _cached_capabilities is not None:
        return _cached_capabilities
    with _capabilities_lock:
        if _cached_capabilities is not None:
            return _cached_capabilities
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                _cached_capabilities = CapabilitiesResponse(
                    has_gpu=False, gpu_count=0, gpu_names=[]
                )
                return _cached_capabilities
            names = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            _cached_capabilities = CapabilitiesResponse(
                has_gpu=len(names) > 0, gpu_count=len(names), gpu_names=names
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            _cached_capabilities = CapabilitiesResponse(
                has_gpu=False, gpu_count=0, gpu_names=[]
            )
    return _cached_capabilities


def reset_cache() -> None:
    """Forget the cached detection so the next request re-detects GPUs."""
    global _cached_capabilities
    with _capabilities_lock:
        _cached_capabilities = None


@router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> Any:
    """Return the system's GPU capabilities.

    Returns:
        A CapabilitiesResponse indicating GPU presence, count, and model names.
    """
    return _detect_gpu()
