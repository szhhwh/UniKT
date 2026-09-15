"""Task status state machine: atomic compare-and-set status transitions.

Every status write goes through :func:`transition`, a filtered UPDATE that
only lands when the row is still in an accepted prior state — so two racing
writers can't both commit. Uses ``logging`` directly (not ``utils.core``) to
stay importable under ``pixi run web-dev`` (no repo-root PYTHONPATH).
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

from sqlalchemy.orm import Session

from services import event_bus

logger = logging.getLogger(__name__)

# Allowed experiment-task transitions; anything else is rejected (loud).
TASK_TRANSITIONS: set[tuple[str, str]] = {
    ("pending", "running"),
    ("pending", "failed"),
    ("pending", "stopped"),
    ("running", "stopping"),
    ("running", "completed"),
    ("running", "failed"),
    ("stopping", "stopped"),
    ("stopping", "failed"),
    ("running", "interrupted"),
    ("stopping", "interrupted"),
    ("interrupted", "stopped"),
    ("interrupted", "completed"),
    ("interrupted", "failed"),
    # Re-queue an in-flight task for re-run on backend restart.
    ("running", "pending"),
    ("stopping", "pending"),
    ("interrupted", "pending"),
}

# Preprocess: no queue, but shares the interrupted/recover lifecycle.
PREPROCESS_TRANSITIONS: set[tuple[str, str]] = {
    ("running", "stopping"),
    ("running", "completed"),
    ("running", "failed"),
    ("running", "stopped"),
    ("stopping", "stopped"),
    ("stopping", "failed"),
    ("running", "interrupted"),
    ("stopping", "interrupted"),
    ("interrupted", "completed"),
    ("interrupted", "failed"),
    ("interrupted", "stopped"),
}

_TABLES: dict[str, set[tuple[str, str]]] = {
    "tasks": TASK_TRANSITIONS,
    "preprocess_tasks": PREPROCESS_TRANSITIONS,
}


def _allowed(model: type, frm: str, to: str) -> bool:
    table = _TABLES.get(getattr(model, "__tablename__", ""), TASK_TRANSITIONS)
    return (frm, to) in table


def transition(
    session: Session,
    model: type[Any],
    task_id: int,
    frm: str | Collection[str],
    to: str,
    **fields: Any,
) -> bool:
    """Atomically move a row from an accepted prior status to ``to`` via CAS.

    Args:
        session: Open SQLAlchemy session (committed here).
        model: ORM class (``Task`` or ``PreprocessTask``).
        task_id: Row primary key.
        frm: Expected current status, or a collection of accepted priors.
        to: Target status.
        **fields: Extra column values to set alongside the status.

    Returns:
        True if exactly one row was updated, else False.
    """
    frms = [frm] if isinstance(frm, str) else list(frm)
    for f in frms:
        if not _allowed(model, f, to):
            logger.warning(
                "Invalid %s transition %s -> %s rejected",
                model.__name__,
                f,
                to,
            )
            return False

    values: dict[Any, Any] = {"status": to, **fields}
    stmt = session.query(model).filter(model.id == task_id)
    if len(frms) == 1:
        stmt = stmt.filter(model.status == frms[0])
    else:
        stmt = stmt.filter(model.status.in_(frms))
    updated = stmt.update(values, synchronize_session=False)
    session.commit()
    if updated == 1:
        etype = (
            "preprocess_status"
            if getattr(model, "__tablename__", "") == "preprocess_tasks"
            else "task_status"
        )
        event_bus.publish({"type": etype, "id": task_id, "status": to})
        return True
    return False
