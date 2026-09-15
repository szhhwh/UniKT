"""Tasks router — CRUD and queue management for experiment tasks.

Provides endpoints to create, list, get, stop, kill, delete, and resize tasks,
as well as managing the task execution queue (list and reorder).
"""

import json
import logging
from datetime import datetime
from typing import Any

from config import SEARCH_TASK_MARKER, TASK_LOGS_DIR
from database import SessionLocal
from dependencies import get_line_cache, get_process_manager
from errors import AppError
from fastapi import APIRouter, Depends
from models import Task
from pagination import Page, Params
from pydantic import BaseModel
from schemas import TaskCreate, TaskResponse
from services.line_render import LineRenderCache
from services.process_manager import ProcessManager
from services.python_env import EnvironmentNotConfigured
from services.task_lifecycle import (
    delete_task_handler,
    kill_task_handler,
    stop_task_handler,
)
from services.task_state import transition
from sqlalchemy import desc, select

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

logger = logging.getLogger(__name__)

# optuna special keys steer _build_cli_args onto the search script with paths
# only /api/search is allowed to choose. Strip them from direct task creation
# so a client cannot point --optuna_config/--output_dir at arbitrary
# filesystem locations.
_RESERVED_SEARCH_KEYS = frozenset(
    {"task_kind", "optuna_config_path", "output_dir", "metric"}
)


@router.post("", response_model=TaskResponse, status_code=201)
def create_task(
    body: TaskCreate, pm: ProcessManager = Depends(get_process_manager)
) -> Any:
    """Create a new experiment task and enqueue it for execution.

    Args:
        body: The task creation request.
        pm: Injected ProcessManager singleton.

    Returns:
        The created Task record.

    Raises:
        AppError: 500 if the task failed to launch.
    """
    params = {k: v for k, v in body.params.items() if k not in _RESERVED_SEARCH_KEYS}
    with SessionLocal() as session:
        dataset_name = params.get("dataset", "")
        task = Task(
            name=body.name,
            command="",
            model_name=body.model_name,
            dataset_name=dataset_name,
            env_type="",
            env_name="",
            status="pending",
            tags="[]",
            # Compact separators: no nested param value can then forge the
            # LIKE marker that classifies a row as a search task.
            extra_params=json.dumps(params, separators=(",", ":")),
            gpu_request=body.gpu,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

    try:
        pm.launch_task(
            task_id=task_id,
            model_name=body.model_name,
            params=params,
            env_id=body.env_id,
            custom_python_path=body.custom_python_path,
        )
    except EnvironmentNotConfigured:
        with SessionLocal() as session:
            transition(
                session,
                Task,
                task_id,
                "pending",
                "failed",
                finished_at=datetime.now(),
            )
        raise
    except Exception:
        logger.exception("Failed to launch task %s (%s)", task_id, body.model_name)
        with SessionLocal() as session:
            transition(
                session,
                Task,
                task_id,
                "pending",
                "failed",
                finished_at=datetime.now(),
            )
        raise AppError("task_launch_failed", 500)

    with SessionLocal() as session:
        task = session.get(Task, task_id)
        return task


@router.get("", response_model=Page[TaskResponse])
def list_tasks(
    status: str | None = None,
    params: Params = Depends(),
) -> Any:
    """List tasks with optional status filter and pagination.

    Active tasks (null finished_at) appear first, then most recently
    finished tasks.

    Args:
        status: Optional status string filter.
        params: Pagination parameters.

    Returns:
        A paginated Page of TaskResponse items.
    """
    from fastapi_pagination.ext.sqlalchemy import paginate

    with SessionLocal() as session:
        stmt = select(Task).order_by(
            Task.finished_at.is_(None).desc(),
            desc(Task.finished_at),
        )
        # Exclude optuna search tasks (owned by the search router) so the training
        # list only shows train.py runs.
        stmt = stmt.where(Task.extra_params.notlike(SEARCH_TASK_MARKER))
        if status:
            stmt = stmt.where(Task.status == status)
        return paginate(session, stmt, params=params)


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: int) -> Any:
    """Return a single task by its ID.

    Args:
        task_id: The task identifier.

    Returns:
        The Task record.

    Raises:
        AppError: 404 if the task does not exist.
    """
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            raise AppError("task_not_found", 404)
        return task


@router.post("/{task_id}/stop")
def stop_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)) -> Any:
    """Request a graceful stop of a running task.

    Raises:
        AppError: 400 if the task cannot be stopped.
    """
    stop_task_handler(pm, task_id)
    return {"status": "stopping"}


@router.post("/{task_id}/kill")
def kill_task(task_id: int, pm: ProcessManager = Depends(get_process_manager)) -> Any:
    """Force-kill a running task.

    Raises:
        AppError: 400 if the task cannot be killed.
    """
    kill_task_handler(pm, task_id)
    return {"status": "killed"}


@router.delete("/{task_id}")
def delete_task(
    task_id: int,
    pm: ProcessManager = Depends(get_process_manager),
    cache: LineRenderCache = Depends(get_line_cache),
) -> Any:
    """Delete a task and its associated logs.

    Raises:
        AppError: 404 if the task does not exist,
            400 if the task is still active (running or stopping).
    """
    return delete_task_handler(pm, cache, task_id, TASK_LOGS_DIR)


@router.get("/queue/list")
def get_queue(pm: ProcessManager = Depends(get_process_manager)) -> Any:
    """Return the ordered task execution queue with task details.

    Args:
        pm: Injected ProcessManager singleton.

    Returns:
        A list of task detail dicts in queue order.
    """
    task_ids = pm.get_queue()
    with SessionLocal() as session:
        tasks = (
            session.query(Task).filter(Task.id.in_(task_ids)).all() if task_ids else []
        )
        order = {tid: i for i, tid in enumerate(task_ids)}
        tasks.sort(key=lambda t: order.get(t.id, 999))
        return [
            {
                "id": t.id,
                "name": t.name,
                "model_name": t.model_name,
                "dataset_name": t.dataset_name,
                "env_name": t.env_name,
                "status": t.status,
                "gpu_request": t.gpu_request,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ]


class ReorderRequest(BaseModel):
    """Request model for reordering the task queue.

    Attributes:
        task_ids: The desired order of task IDs in the queue.
    """

    task_ids: list[int]


@router.put("/queue/reorder")
def reorder_queue(
    body: ReorderRequest, pm: ProcessManager = Depends(get_process_manager)
) -> Any:
    """Reorder the task execution queue.

    Args:
        body: The reorder request with the desired task ID order.
        pm: Injected ProcessManager singleton.

    Returns:
        A dict with ``ok`` set to ``True``.
    """
    pm.reorder_queue(body.task_ids)
    return {"ok": True}


class CommandPreviewRequest(BaseModel):
    """Request model for previewing a task's CLI invocation.

    Attributes:
        model_name: The model to train.
        params: Flat parameter dict (field_name -> value).
    """

    model_name: str
    params: dict


class CommandPreviewResponse(BaseModel):
    """Response model for command preview.

    Attributes:
        command: The full CLI invocation string.
    """

    command: str


@router.post("/preview-command", response_model=CommandPreviewResponse)
def preview_command(
    body: CommandPreviewRequest,
    pm: ProcessManager = Depends(get_process_manager),
) -> Any:
    """Return the CLI command that would be executed for the given params."""
    command = pm.preview_command(body.model_name, body.params)
    return CommandPreviewResponse(command=command)
