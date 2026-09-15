"""Search router — Optuna hyperparameter search tasks.

Reuses the training ProcessManager (GPU queue, PTY logs, state machine, crash
recovery) by stamping each task with ``task_kind="optuna"`` so the command
builder dispatches ``optuna_search.py`` instead of ``train.py``. Search tasks
share the ``tasks`` table and ``task_status`` event stream with training tasks;
this router only adds the optuna-specific create/preview path and study-db read
endpoints (trial progress, optuna-dashboard command).
"""

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    SEARCH_CONFIGS_DIR,
    SEARCH_RUNS_DIR,
    SEARCH_TASK_MARKER,
    TASK_LOGS_DIR,
)
from database import SessionLocal
from dependencies import get_line_cache, get_process_manager
from errors import AppError
from fastapi import APIRouter, Depends
from models import Task
from pagination import Page, Params
from pydantic import BaseModel
from schemas import (
    SearchCreate,
    SearchStudyPathResponse,
    SearchStudyResponse,
    TaskResponse,
)
from services.line_render import LineRenderCache
from services.optuna_config_writer import write_optuna_config
from services.process_manager import ProcessManager
from services.python_env import EnvironmentNotConfigured
from services.study_reader import read_study
from services.task_lifecycle import (
    delete_task_handler,
    kill_task_handler,
    stop_task_handler,
)
from services.task_state import transition
from sqlalchemy import desc, select

router = APIRouter(prefix="/api/search", tags=["search"])

logger = logging.getLogger(__name__)

# Optuna direction is metric-driven (mirrors utils.optuna_utils.direction_for_metric,
# duplicated here so the web env need not import optuna).
_METRIC_DIRECTION = {
    "auc": "maximize",
    "acc": "maximize",
    "rmse": "minimize",
    "loss": "minimize",
}


class SearchPreviewRequest(BaseModel):
    """Request model for previewing a search's CLI invocation.

    Attributes:
        model_name: The model to search over.
        dataset: The dataset to search on.
        runconfig_params: Flat RunConfig knobs used as the per-trial base config.
        optuna_config: Optuna study knobs (metric/n_trials/sampler/...).
    """

    model_name: str
    dataset: str
    runconfig_params: dict = {}
    optuna_config: dict = {}


class SearchPreviewResponse(BaseModel):
    """Response model for search command preview."""

    command: str


def _write_search_yaml(optuna_config: dict, task_id: int | None) -> tuple[str, str]:
    """Persist the optuna config YAML and return (config_path, output_dir).

    A real task uses ``<task_id>``-named artifacts; previews share a fixed
    ``_preview`` pair so repeated 200ms-debounced previews overwrite harmlessly.
    """
    if task_id is not None:
        config_path = SEARCH_CONFIGS_DIR / f"{task_id}.yaml"
        output_dir = str(SEARCH_RUNS_DIR / f"web_{task_id}")
    else:
        config_path = SEARCH_CONFIGS_DIR / "_preview.yaml"
        output_dir = str(SEARCH_RUNS_DIR / "_preview")
    write_optuna_config(optuna_config, config_path)
    return str(config_path), output_dir


def _build_search_params(
    runconfig_params: dict,
    dataset: str,
    optuna_config: dict,
    task_id: int | None = None,
) -> dict:
    """Assemble the params dict stored in ``Task.extra_params``.

    The ProcessManager reads this back to rebuild the command, so it must carry
    both the RunConfig knobs (routed into ``--node.field`` flags) and the
    optuna-only keys the command builder special-cases.
    """
    metric = optuna_config.get("metric", "auc")
    config_path, output_dir = _write_search_yaml(optuna_config, task_id)
    return {
        **runconfig_params,
        "dataset": dataset,
        "task_kind": "optuna",
        "metric": metric,
        "optuna_config_path": config_path,
        "output_dir": output_dir,
    }


def _cleanup_search_artifacts(task_id: int) -> None:
    """Remove the persisted optuna YAML for a task (search results are kept)."""
    config_path = SEARCH_CONFIGS_DIR / f"{task_id}.yaml"
    if config_path.is_file():
        with contextlib.suppress(OSError):
            config_path.unlink()


def _load_extra_params(task_id: int) -> dict | None:
    """Return the parsed ``extra_params`` for a task, or None if it is absent."""
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            return None
        return json.loads(task.extra_params or "{}")


def _require_search_task(task_id: int) -> None:
    """Raise 404 unless the id addresses a search task.

    Search and training tasks share the ``tasks`` table; restrict these
    endpoints to rows carrying the optuna marker so training ids are not
    operable here (deleted, killed, or misread as an empty study).
    """
    with SessionLocal() as session:
        found = session.execute(
            select(Task.id).where(
                Task.id == task_id,
                Task.extra_params.like(SEARCH_TASK_MARKER),
            )
        ).first()
    if found is None:
        raise AppError("task_not_found", 404)


@router.post("", response_model=TaskResponse, status_code=201)
def create_search(
    body: SearchCreate, pm: ProcessManager = Depends(get_process_manager)
) -> Any:
    """Create a hyperparameter search task and enqueue it for execution."""
    name = body.name or f"{body.model_name}_{body.dataset}_search"
    with SessionLocal() as session:
        task = Task(
            name=name,
            command="",
            model_name=body.model_name,
            dataset_name=body.dataset,
            env_type="",
            env_name="",
            status="pending",
            tags="[]",
            # Marker-bearing placeholder: a failure before params are built
            # still classifies the row as a search task instead of leaking
            # into the training list.
            extra_params=json.dumps({"task_kind": "optuna"}),
            gpu_request=body.gpu,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

    try:
        params = _build_search_params(
            body.runconfig_params, body.dataset, body.optuna_config, task_id=task_id
        )
    except Exception:
        logger.exception("Failed to build search config for task %s", task_id)
        with SessionLocal() as session:
            transition(
                session, Task, task_id, "pending", "failed", finished_at=datetime.now()
            )
        raise AppError("task_launch_failed", 500)

    # Persist the assembled params (carrying the task_kind marker) BEFORE launch
    # so a launch failure still leaves a correctly classified row — otherwise a
    # failed search would leak into the training list as an unmarked ghost task.
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        task.extra_params = json.dumps(params)
        session.commit()

    try:
        pm.launch_task(
            task_id=task_id,
            model_name=body.model_name,
            params=params,
            env_id=body.env_id,
            custom_python_path=body.custom_python_path,
        )
    except EnvironmentNotConfigured:
        _cleanup_search_artifacts(task_id)
        with SessionLocal() as session:
            transition(
                session, Task, task_id, "pending", "failed", finished_at=datetime.now()
            )
        raise
    except Exception:
        logger.exception("Failed to launch search %s (%s)", task_id, body.model_name)
        _cleanup_search_artifacts(task_id)
        with SessionLocal() as session:
            transition(
                session, Task, task_id, "pending", "failed", finished_at=datetime.now()
            )
        raise AppError("task_launch_failed", 500)

    with SessionLocal() as session:
        return session.get(Task, task_id)


@router.post("/preview-command", response_model=SearchPreviewResponse)
def preview_command(
    body: SearchPreviewRequest, pm: ProcessManager = Depends(get_process_manager)
) -> Any:
    """Return the CLI command that would be executed for the given search config."""
    params = _build_search_params(
        body.runconfig_params, body.dataset, body.optuna_config, task_id=None
    )
    return SearchPreviewResponse(command=pm.preview_command(body.model_name, params))


@router.get("", response_model=Page[TaskResponse])
def list_searches(status: str | None = None, params: Params = Depends()) -> Any:
    """List search tasks with optional status filter and pagination.

    Active tasks (null finished_at) appear first, then most recently finished.
    """
    from fastapi_pagination.ext.sqlalchemy import paginate

    with SessionLocal() as session:
        stmt = (
            select(Task)
            .where(Task.extra_params.like(SEARCH_TASK_MARKER))
            .order_by(
                Task.finished_at.is_(None).desc(),
                desc(Task.finished_at),
            )
        )
        if status:
            stmt = stmt.where(Task.status == status)
        return paginate(session, stmt, params=params)


@router.get("/{task_id}", response_model=TaskResponse)
def get_search(task_id: int) -> Any:
    """Return a single search task by its ID.

    Raises:
        AppError: 404 if the task does not exist or is not a search task.
    """
    _require_search_task(task_id)
    with SessionLocal() as session:
        task = session.get(Task, task_id)
        if not task:
            raise AppError("task_not_found", 404)
        return task


@router.post("/{task_id}/stop")
def stop_search(task_id: int, pm: ProcessManager = Depends(get_process_manager)) -> Any:
    """Request a graceful stop of a running search."""
    _require_search_task(task_id)
    stop_task_handler(pm, task_id)
    return {"status": "stopping"}


@router.post("/{task_id}/kill")
def kill_search(task_id: int, pm: ProcessManager = Depends(get_process_manager)) -> Any:
    """Force-kill a running search."""
    _require_search_task(task_id)
    kill_task_handler(pm, task_id)
    return {"status": "killed"}


@router.delete("/{task_id}")
def delete_search(
    task_id: int,
    pm: ProcessManager = Depends(get_process_manager),
    cache: LineRenderCache = Depends(get_line_cache),
) -> Any:
    """Delete a search task, its log, and its persisted optuna YAML.

    The search output directory (study.db, trial subdirs) is preserved.
    """
    _require_search_task(task_id)
    return delete_task_handler(
        pm, cache, task_id, TASK_LOGS_DIR, post_cleanup=_cleanup_search_artifacts
    )


@router.get("/{task_id}/trials", response_model=SearchStudyResponse)
def get_search_trials(task_id: int) -> Any:
    """Return live trial progress read from the search's ``study.db``.

    Renders an empty summary when the DB is not ready yet (search not started or
    optuna has not created it). Raises AppError 404 if the task does not exist
    or is not a search task.
    """
    _require_search_task(task_id)
    extra = _load_extra_params(task_id)
    if extra is None:
        raise AppError("task_not_found", 404)
    output_dir = extra.get("output_dir")
    direction = _METRIC_DIRECTION.get(extra.get("metric", "auc"), "maximize")
    empty = SearchStudyResponse(
        total=0, completed=0, running=0, pruned=0, failed=0, direction=direction
    )
    if not output_dir:
        return empty
    result = read_study(Path(output_dir) / "study.db", direction=direction)
    return empty if result is None else result


@router.get("/{task_id}/study-db", response_model=SearchStudyPathResponse)
def get_study_db_path(task_id: int) -> Any:
    """Return the ``study.db`` path and a copy-paste optuna-dashboard command.

    Raises AppError 404 if the task does not exist or is not a search task.
    """
    _require_search_task(task_id)
    extra = _load_extra_params(task_id)
    if extra is None:
        raise AppError("task_not_found", 404)
    output_dir = extra.get("output_dir")
    if not output_dir:
        return SearchStudyPathResponse()
    study_db = Path(output_dir) / "study.db"
    abs_path = str(study_db.resolve()) if study_db.exists() else str(study_db)
    return SearchStudyPathResponse(
        study_db_path=str(study_db),
        dashboard_command=f"optuna-dashboard sqlite:///{abs_path}",
        exists=study_db.is_file(),
    )
