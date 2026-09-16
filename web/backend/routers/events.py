"""Events router — SSE stream of task/preprocess status changes.

On connect, sends a snapshot of every task and preprocess task status, then
streams incremental status events published via the event bus.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from database import SessionLocal
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models import PreprocessTask, Task
from services import event_bus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"])

KEEPALIVE_SECONDS = 15.0


def _status_snapshot() -> list[dict]:
    """Snapshot every task and preprocess status for a new SSE subscriber."""
    with SessionLocal() as session:
        return [
            {"type": "task_status", "id": t.id, "status": t.status, "pid": t.pid}
            for t in session.query(Task).all()
        ] + [
            {"type": "preprocess_status", "id": p.id, "status": p.status}
            for p in session.query(PreprocessTask).all()
        ]


@router.get("/api/events")
async def events() -> Any:
    """Server-sent events stream of status changes."""

    async def gen() -> AsyncIterator[str]:
        # Subscribe before snapshotting: events published in between are queued
        # rather than lost. Duplicate snapshots are harmless — clients overwrite
        # by id; a missed terminal status is not.
        q = event_bus.subscribe()
        try:
            # The full-table read runs in a worker thread so the event loop
            # stays free for the other SSE/HTTP/WS clients.
            for event in await asyncio.to_thread(_status_snapshot):
                yield f"data: {json.dumps(event)}\n\n"

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")
