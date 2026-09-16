"""Preprocess router — data download/processing tasks.

Provides CRUD endpoints for launching and managing preprocess tasks (download
or process actions), including WebSocket log streaming.
"""

import logging
from typing import Any

from config import PREPROCESS_LOGS_DIR
from dependencies import get_line_cache, get_preprocess_manager
from errors import AppError
from fastapi import (
    APIRouter,
    Depends,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel
from services.line_render import LineRenderCache
from services.log_reader import read_log_lines, stream_log_lines
from services.preprocess_manager import PreprocessManager
from services.python_env import EnvironmentNotConfigured

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])


class PreprocessStartRequest(BaseModel):
    """Request model for starting a preprocess task.

    Attributes:
        action: The action to perform (``download`` or ``process``).
        dataset: Name of the dataset to preprocess.
        params: Additional parameters for the action.
        env_id: Optional environment identifier.
        custom_python_path: Optional custom Python interpreter path.
    """

    action: str
    dataset: str
    params: dict = {}
    env_id: str | None = None
    custom_python_path: str | None = None


@router.post("", status_code=201)
def start_preprocess(
    body: PreprocessStartRequest,
    pm: PreprocessManager = Depends(get_preprocess_manager),
) -> Any:
    """Start a new preprocess task (download or process).

    Args:
        body: The preprocess start request.
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with the task ``id``, ``command``, ``status``, and ``started_at``.

    Raises:
        AppError: 400 if the action or dataset is invalid.
    """
    if body.action not in ("download", "process"):
        raise AppError("preprocess_action_invalid")
    if not body.dataset:
        raise AppError("preprocess_dataset_required")
    try:
        task = pm.start(
            body.action, body.dataset, body.params, body.env_id, body.custom_python_path
        )
    except EnvironmentNotConfigured:
        raise
    except KeyError:
        raise AppError("preprocess_schema_unavailable", 503)
    return {
        "id": task.id,
        "command": task.command,
        "status": task.status,
        "started_at": task.started_at.isoformat() if task.started_at else None,
    }


@router.post("/preview")
def preview_preprocess(
    body: PreprocessStartRequest,
    pm: PreprocessManager = Depends(get_preprocess_manager),
) -> Any:
    """Preview the command for a preprocess config without launching.

    Args:
        body: The preprocess start request (action/dataset/params/env).
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with the preview ``command`` string.

    Raises:
        AppError: 400 if action/dataset invalid or env not configured;
            503 if the preprocess schema is unavailable.
    """
    if body.action not in ("download", "process"):
        raise AppError("preprocess_action_invalid")
    if not body.dataset:
        raise AppError("preprocess_dataset_required")
    try:
        command = pm.preview_command(body.action, body.dataset, body.params)
    except EnvironmentNotConfigured:
        raise
    except KeyError:
        raise AppError("preprocess_schema_unavailable", 503)
    return {"command": command}


@router.get("")
def list_preprocess(pm: PreprocessManager = Depends(get_preprocess_manager)) -> Any:
    """List all preprocess tasks.

    Args:
        pm: Injected PreprocessManager singleton.

    Returns:
        A list of task dicts with ``id``, ``command``, ``status``, etc.
    """
    tasks = []
    for t in pm.list_all():
        tasks.append(
            {
                "id": t.id,
                "command": t.command,
                "status": t.status,
                "exit_code": t.exit_code,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "finished_at": t.finished_at.isoformat() if t.finished_at else None,
            }
        )
    return tasks


@router.get("/{task_id}")
def get_preprocess(
    task_id: int, pm: PreprocessManager = Depends(get_preprocess_manager)
) -> Any:
    """Return details for a specific preprocess task.

    Args:
        task_id: The preprocess task identifier.
        pm: Injected PreprocessManager singleton.

    Returns:
        A task dict with ``id``, ``command``, ``status``, etc.

    Raises:
        AppError: 404 if the task does not exist.
    """
    task = pm.get(task_id)
    if not task:
        raise AppError("preprocess_not_found", 404)
    return {
        "id": task.id,
        "command": task.command,
        "status": task.status,
        "exit_code": task.exit_code,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
    }


@router.post("/{task_id}/stop")
def stop_preprocess(
    task_id: int, pm: PreprocessManager = Depends(get_preprocess_manager)
) -> Any:
    """Stop a running preprocess task.

    Args:
        task_id: The preprocess task identifier.
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with ``status`` set to ``stopping``.

    Raises:
        AppError: 400 if the task cannot be stopped.
    """
    if not pm.stop(task_id):
        raise AppError("cannot_stop_preprocess")
    return {"status": "stopping"}


@router.delete("/{task_id}")
def delete_preprocess(
    task_id: int,
    pm: PreprocessManager = Depends(get_preprocess_manager),
) -> Any:
    """Delete a preprocess task and its log.

    Args:
        task_id: The preprocess task identifier.
        pm: Injected PreprocessManager singleton.

    Returns:
        A dict with ``status`` set to ``deleted``.

    Raises:
        AppError: 400 if the task cannot be deleted.
    """
    if not pm.delete(task_id):
        raise AppError("cannot_delete_preprocess")
    return {"status": "deleted"}


@router.get("/{task_id}/logs")
def get_preprocess_logs(
    task_id: int,
    cache: LineRenderCache = Depends(get_line_cache),
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
) -> Any:
    """Fetch rendered preprocess log lines.

    Args:
        task_id: The preprocess task identifier.
        cache: Injected LineRenderCache singleton.
        offset: Starting line index (0-based).
        limit: Max number of lines to return (1-5000).

    Returns:
        ``{"lines": [...], "total": int}``.
    """
    return read_log_lines(PREPROCESS_LOGS_DIR / f"{task_id}.log", cache, offset, limit)


@router.websocket("/{task_id}/logs/stream")
async def stream_preprocess_logs(
    websocket: WebSocket,
    task_id: int,
    cache: LineRenderCache = Depends(get_line_cache),
    from_line: int = Query(0, ge=0),
    pm: PreprocessManager = Depends(get_preprocess_manager),
) -> None:
    """Stream preprocess task logs live over a WebSocket connection.

    Args:
        websocket: The WebSocket connection.
        task_id: The preprocess task identifier.
        cache: Injected LineRenderCache singleton.
        from_line: Line index the client already has.
        pm: Injected PreprocessManager singleton.
    """
    await websocket.accept()
    task = pm.get(task_id)
    if not task:
        await websocket.send_json({"type": "error", "content": "task_not_found"})
        await websocket.close()
        return

    def check_alive() -> bool:
        t = pm.get(task_id)
        return t is not None and t.status in ("running", "stopping")

    try:
        await stream_log_lines(
            PREPROCESS_LOGS_DIR / f"{task_id}.log",
            websocket,
            cache,
            check_alive=check_alive,
            from_line=from_line,
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Preprocess WebSocket stream error")
