"""Settings API router — application settings management.

Provides endpoints for reading and updating global settings (per-GPU task
slots, default environment, setup status), as well as managing the default
Python environment selection.
"""

from typing import Any

from dependencies import get_process_manager, get_settings_manager
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from services.process_manager import ProcessManager
from services.settings_manager import SettingsManager

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsResponse(BaseModel):
    """Response model for current settings.

    Attributes:
        gpu_slots: Concurrent task slots per GPU (or the CPU lane).
    """

    gpu_slots: int


class SettingsUpdate(BaseModel):
    """Request model for updating settings.

    Attributes:
        gpu_slots: New per-GPU slot count.
    """

    gpu_slots: int


@router.get("")
def get_settings(sm: SettingsManager = Depends(get_settings_manager)) -> Any:
    """Return current application settings.

    Args:
        sm: Injected SettingsManager singleton.

    Returns:
        A SettingsResponse with the current gpu_slots value.
    """
    return SettingsResponse(gpu_slots=sm.get_gpu_slots())


@router.put("")
def update_settings(
    body: SettingsUpdate,
    pm: ProcessManager = Depends(get_process_manager),
    sm: SettingsManager = Depends(get_settings_manager),
) -> Any:
    """Update application settings.

    Args:
        body: The settings update request.
        pm: Injected ProcessManager singleton.
        sm: Injected SettingsManager singleton.

    Returns:
        An updated SettingsResponse.
    """
    value = sm.set_gpu_slots(body.gpu_slots)
    pm.gpu_slots = value
    return SettingsResponse(gpu_slots=value)


class DefaultEnvUpdate(BaseModel):
    """Request model for updating the default Python environment.

    Attributes:
        env_id: The environment identifier to set as default.
        custom_python_path: Optional custom Python interpreter path.
        remember_last_env: Whether to remember the last used environment.
    """

    env_id: str
    custom_python_path: str | None = None
    remember_last_env: bool | None = None


@router.get("/default-env")
def get_default_env(sm: SettingsManager = Depends(get_settings_manager)) -> Any:
    """Return the current default environment configuration.

    Args:
        sm: Injected SettingsManager singleton.

    Returns:
        A dict with ``default_env_id``, ``custom_python_path``, and
        ``remember_last_env``.
    """
    return {
        "default_env_id": sm.get_default_env(),
        "custom_python_path": sm.get_custom_python_path(),
        "remember_last_env": sm.get_remember_last_env(),
    }


@router.post("/default-env")
def set_default_env(
    body: DefaultEnvUpdate, sm: SettingsManager = Depends(get_settings_manager)
) -> Any:
    """Set the default Python environment.

    Args:
        body: The default environment update request.
        sm: Injected SettingsManager singleton.

    Returns:
        A dict with ``ok`` and the new ``default_env_id``.
    """
    sm.set_default_env(body.env_id, body.custom_python_path)
    if body.remember_last_env is not None:
        sm.set_remember_last_env(body.remember_last_env)
    return {"ok": True, "default_env_id": body.env_id}


@router.get("/initialized")
def check_initialized(sm: SettingsManager = Depends(get_settings_manager)) -> Any:
    """Check whether the initial setup has been completed.

    Args:
        sm: Injected SettingsManager singleton.

    Returns:
        A dict with ``initialized`` boolean.
    """
    return {"initialized": sm.is_setup_completed()}
