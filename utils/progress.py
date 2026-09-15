"""Shared Rich progress bar factory."""

import sys

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column


def create_progress() -> Progress:
    """Create a Rich Progress bar with the project-standard style.

    Returns:
        A configured ``Progress`` instance ready for ``add_task()``.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(table_column=Column(justify="right")),
        TimeRemainingColumn(),
        expand=True,
    )


def resolve_progress(mode: str) -> bool:
    """Return whether Rich progress rendering is enabled for ``mode``.

    Args:
        mode: Progress mode — ``"auto"``, ``"rich"``, or ``"none"``.

    Returns:
        True if progress should be rendered.

    Raises:
        ValueError: The mode is not one of the supported values.
    """
    if mode == "rich":
        return True
    if mode == "none":
        return False
    if mode == "auto":
        return sys.stdout.isatty()
    raise ValueError(f"Unknown progress mode: {mode!r}")
