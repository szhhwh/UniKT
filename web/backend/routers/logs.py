"""Logs router — task log retrieval and WebSocket streaming (line-oriented)."""

import logging
from typing import Any

from config import TASK_LOGS_DIR
from database import SessionLocal
from dependencies import get_line_cache
from errors import AppError
from fastapi import (
    APIRouter,
    Depends,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from models import Task
from services.line_render import LineRenderCache
from services.log_reader import read_log_lines, stream_log_lines

logger = logging.getLogger(__name__)

router = APIRouter(tags=["logs"])


@router.get("/api/tasks/{task_id}/logs")
def get_logs(
    task_id: int,
    cache: LineRenderCache = Depends(get_line_cache),
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
) -> Any:
    """Fetch rendered log lines for a task.

    Args:
        task_id: The task identifier.
        cache: Injected LineRenderCache singleton.
        offset: Starting line index (0-based).
        limit: Max number of lines to return (1-5000).

    Returns:
        ``{"lines": [...], "total": int}``.

    Raises:
        AppError: 404 if the task does not exist.
    """
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            raise AppError("task_not_found", 404)
    return read_log_lines(TASK_LOGS_DIR / f"{task_id}.log", cache, offset, limit)


@router.websocket("/api/tasks/{task_id}/logs/stream")
async def stream_logs(
    websocket: WebSocket,
    task_id: int,
    cache: LineRenderCache = Depends(get_line_cache),
    from_line: int = Query(0, ge=0),
) -> None:
    """Stream rendered task log lines live as incremental patches.

    Args:
        websocket: The WebSocket connection.
        task_id: The task identifier.
        cache: Injected LineRenderCache singleton.
        from_line: Line index the client already has; the stream aligns from
            here and then pushes only new/changed lines.
    """
    await websocket.accept()
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            await websocket.send_json({"type": "error", "content": "task_not_found"})
            await websocket.close()
            return

    def check_alive() -> bool:
        with SessionLocal() as session:
            t = session.get(Task, task_id)
            if not t:
                return False
            # A queued task has no pid yet but its output is still to come;
            # reporting it dead closes the stream before it ever starts.
            if t.status == "pending":
                return True
            return t.pid is not None and t.status in ("running", "stopping")

    try:
        await stream_log_lines(
            TASK_LOGS_DIR / f"{task_id}.log",
            websocket,
            cache,
            check_alive=check_alive,
            from_line=from_line,
        )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket stream error")
