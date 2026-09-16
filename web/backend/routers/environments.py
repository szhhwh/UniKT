"""Environments router — Python environment discovery and health checks.

Lists available Python environments and runs health checks (Python + PyTorch
availability) on a given environment.
"""

from typing import Any

from dependencies import get_python_env_manager
from fastapi import APIRouter, Depends
from schemas import EnvHealthCheckRequest
from services.python_env import PythonEnvManager

router = APIRouter(prefix="/api/environments", tags=["environments"])


@router.get("")
def list_environments(mgr: PythonEnvManager = Depends(get_python_env_manager)) -> Any:
    """List all discovered Python environments (pixi, conda, custom).

    Args:
        mgr: Injected PythonEnvManager singleton.

    Returns:
        A list of EnvironmentInfo objects.
    """
    return mgr.discover()


@router.post("/health-check")
async def health_check(
    body: EnvHealthCheckRequest,
    mgr: PythonEnvManager = Depends(get_python_env_manager),
) -> Any:
    """Run a health check against a specific Python environment.

    Args:
        body: Request containing the environment ID and optional custom Python path.
        mgr: Injected PythonEnvManager singleton.

    Returns:
        A dict with Python availability, version, Torch availability, and errors.
    """
    return await mgr.health_check(body.env_id, body.custom_python_path)
