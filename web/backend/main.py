"""FastAPI application entry point for the KT Experiment Manager.

Configures the FastAPI app with CORS, error handling, pagination, and registers all API routers.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http.client import responses

from config import read_env_file_value
from database import init_db
from errors import AppError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from fastapi_pagination.api import set_page
from fastapi_problem.error import Problem
from fastapi_problem.handler import (
    ExceptionHandler,
    add_exception_handler,
    new_exception_handler,
)
from middleware import MessageMiddleware
from pagination import Page
from routers import (
    capabilities,
    datasets,
    environments,
    events,
    gpu,
    logs,
    preprocess,
    registry,
    resource,
    schemas_api,
    search,
    settings_api,
    tasks,
)
from starlette.requests import Request

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    Initializes the database, creates manager/dependency singletons, and
    recovers interrupted tasks on startup. Shuts down all managers on exit.
    """
    import asyncio

    import dependencies as deps
    from services import app_lock, event_bus

    app_lock.acquire()
    try:
        init_db()
        event_bus.set_loop(asyncio.get_running_loop())
        deps.line_cache = __import__(
            "services.line_render", fromlist=["LineRenderCache"]
        ).LineRenderCache()

        deps.settings_manager = __import__(
            "services.settings_manager", fromlist=["SettingsManager"]
        ).SettingsManager()
        deps.python_env_manager = __import__(
            "services.python_env", fromlist=["PythonEnvManager"]
        ).PythonEnvManager(settings_manager=deps.settings_manager)
        deps.gpu_monitor = __import__(
            "services.gpu_monitor", fromlist=["GpuMonitor"]
        ).GpuMonitor()
        deps.schema_extractor = __import__(
            "services.schema_extractor", fromlist=["SchemaExtractor"]
        ).SchemaExtractor(env_manager=deps.python_env_manager)
        deps.process_manager = __import__(
            "services.process_manager", fromlist=["ProcessManager"]
        ).ProcessManager(
            env_manager=deps.python_env_manager,
            gpu_monitor=deps.gpu_monitor,
            schema_extractor=deps.schema_extractor,
            line_cache=deps.line_cache,
        )
        deps.process_manager.gpu_slots = deps.settings_manager.get_gpu_slots()
        deps.process_manager.recover_tasks()
        deps.preprocess_manager = __import__(
            "services.preprocess_manager", fromlist=["PreprocessManager"]
        ).PreprocessManager(
            env_manager=deps.python_env_manager,
            schema_extractor=deps.schema_extractor,
            line_cache=deps.line_cache,
        )
        deps.preprocess_manager.recover_tasks()
        yield
    finally:
        # Must run even when the lifespan task is cancelled or the server exits
        # with an exception, or running subprocesses keep their GPUs forever.
        for manager in (
            deps.preprocess_manager,
            deps.process_manager,
            deps.gpu_monitor,
        ):
            if manager:
                try:
                    manager.shutdown()
                except Exception:
                    logger.exception("shutdown failed for %s", type(manager).__name__)
        app_lock.release()


app = FastAPI(title="KT Experiment Manager", lifespan=lifespan)


def app_error_handler(
    _eh: ExceptionHandler, _request: Request, exc: AppError
) -> Problem:
    """Map an AppError to a Problem with its code as the type."""
    title = responses.get(exc.status, "Error")
    return Problem(
        title=title, type_=exc.code, status=exc.status, detail=exc.detail or exc.code
    )


eh = new_exception_handler(handlers={AppError: app_error_handler})
add_exception_handler(app, eh)

# The frontend only ever calls same-origin through the Vite proxy, so the
# browser origin is the dev/preview server itself. Restrict cross-origin
# access to those origins: the API is unauthenticated and can launch
# processes, so a random webpage must not be able to drive it.
_frontend_port = (
    os.environ.get("KT_WEB_PORT") or read_env_file_value("KT_WEB_PORT") or "5173"
)
_allowed_origins = [
    f"http://{host}:{_frontend_port}" for host in ("localhost", "127.0.0.1")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Messages"],
)

app.add_middleware(MessageMiddleware)

set_page(Page)
add_pagination(app)

app.include_router(tasks.router)
app.include_router(search.router)
app.include_router(logs.router)
app.include_router(events.router)
app.include_router(environments.router)
app.include_router(schemas_api.router)
app.include_router(registry.router)
app.include_router(gpu.router)
app.include_router(resource.router)
app.include_router(preprocess.router)
app.include_router(datasets.router)
app.include_router(settings_api.router)
app.include_router(capabilities.router)
