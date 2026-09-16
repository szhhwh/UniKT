"""Unified logging module for the KT experiment framework.

This module provides a consistent logging interface for all framework
components. The log level is controlled by the ``LOG_LEVEL`` environment
variable (default: INFO).

A shared file sink can be registered via :func:`add_file_handler`. Entry
scripts (``train.py``, ``evaluate.py``, ``case_analysis.py``, ...) call it
manually with the run directory, so every framework logger writes to the
run's log file alongside the console output. While the sink is registered,
``warnings.warn`` output from the owning process is forwarded to both the
original warning hook and the same file as WARNING records. Forked workers
intentionally keep their own warning path instead of inheriting a shared
file sink, because a regular ``FileHandler`` cannot be rotated safely across
processes.

Usage:
    from utils.core import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("Missing data, using defaults")
    logger.error("Model loading failed")

Environment Variables:
    LOG_LEVEL: Controls logging verbosity (DEBUG, INFO, WARNING, ERROR).
               Default: INFO

Log Format:
    Console: ``[HH:MM:SS][LEVEL][module_name] message`` (RichHandler)
    File:    ``%(asctime)s %(levelname)-8s %(name)s: %(message)s``
"""

import logging
import os
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich.logging import RichHandler

# Global logger cache to avoid duplicate handlers
_loggers: dict = {}

# Shared file sink, appended to every framework logger when set. Forked
# DataLoader workers inherit the handler, but the warning hook is removed in
# the child process so worker warnings do not write to a stale parent sink.
_file_handler: logging.FileHandler | None = None

# Keep captured warning records on a private, non-registered logger. Using the
# stdlib ``py.warnings`` logger directly would mutate host applications'
# levels, handlers, and root propagation settings.
_warning_logger = logging.Logger("py.warnings")
_warning_logger.propagate = False
_warning_previous_showwarning: Callable[..., None] | None = None
_warning_capture_active = False


def _get_log_level_from_env() -> int:
    """Get the log level from the ``LOG_LEVEL`` environment variable.

    Returns:
        A logging level constant (``logging.DEBUG``, ``logging.INFO``, etc.).
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_mapping.get(log_level_str, logging.INFO)


def _all_framework_loggers() -> list[logging.Logger]:
    """Every framework logger the file sink and level updates must reach.

    Module-level loggers captured at import time keep living in the global
    logging registry after :func:`reset_loggers` empties ``_loggers``, so we
    also recover them from there via their RichHandler signature.
    """
    seen: set[int] = set()
    targets: list[logging.Logger] = []
    for lg in _loggers.values():
        if id(lg) not in seen:
            seen.add(id(lg))
            targets.append(lg)
    for lg in logging.Logger.manager.loggerDict.values():
        # Skip PlaceHolder entries; keep only our loggers (RichHandler marker).
        if not isinstance(lg, logging.Logger) or id(lg) in seen:
            continue
        if any(isinstance(h, RichHandler) for h in lg.handlers):
            seen.add(id(lg))
            targets.append(lg)
    return targets


def _detach_from_all(handler: logging.Handler) -> None:
    """Remove ``handler`` from every framework logger (idempotent)."""
    for lg in _all_framework_loggers():
        if handler in lg.handlers:
            lg.removeHandler(handler)


def _detach_warning_handler(handler: logging.Handler) -> None:
    """Detach the private warning sink without touching host loggers."""
    if handler in _warning_logger.handlers:
        _warning_logger.removeHandler(handler)


def _warning_showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Forward a warning to the prior hook and the active file sink."""
    previous = _warning_previous_showwarning
    if previous is not None and previous is not _warning_showwarning:
        previous(message, category, filename, lineno, file, line)

    # The prior hook remains responsible for stderr or an explicitly supplied
    # file. Only the normal warning path is duplicated into the run log.
    if file is None and _warning_capture_active and _file_handler is not None:
        _warning_logger.warning(
            "%s",
            warnings.formatwarning(message, category, filename, lineno, line),
        )


def _install_warning_capture() -> None:
    """Install the warning hook while preserving the current hook."""
    global _warning_capture_active, _warning_previous_showwarning
    if warnings.showwarning is not _warning_showwarning:
        _warning_previous_showwarning = warnings.showwarning
        warnings.showwarning = _warning_showwarning
    _warning_capture_active = True


def _restore_warning_capture() -> None:
    """Restore the prior warning hook when it is still ours."""
    global _warning_capture_active, _warning_previous_showwarning
    if not _warning_capture_active:
        return
    if warnings.showwarning is _warning_showwarning:
        previous = _warning_previous_showwarning
        if previous is not None:
            warnings.showwarning = previous
    _warning_previous_showwarning = None
    _warning_capture_active = False


def _disable_warning_capture_after_fork() -> None:
    """Keep forked workers from writing warnings to the parent's sink."""
    _restore_warning_capture()
    if _file_handler is not None:
        _warning_logger.removeHandler(_file_handler)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_disable_warning_capture_after_fork)


def _write_session_header(handler: logging.FileHandler) -> None:
    """Prefix this session with a separator header.

    evaluate/case_analysis reuse the run_dir, so appending without a banner
    would merge two runs' output; the timestamp + pid keeps them apart.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stream = handler.stream
    assert stream is not None, "file handler stream must be open"
    stream.write(
        f"\n{'=' * 70}\n"
        f"# run.log session start  {timestamp}  pid={os.getpid()}\n"
        f"{'=' * 70}\n"
    )
    handler.flush()


def add_file_handler(log_path: str | Path) -> Path:
    """Attach the shared file sink, writing to ``log_path``.

    Any previously attached file handler is detached and closed first, so a
    second call (e.g. a new run directory in a kfold loop) switches the sink.
    The handler is appended to every framework logger; loggers created
    afterwards pick it up via :func:`get_logger`. :func:`warnings.warn`
    output from this process is forwarded to the prior warning hook and
    captured into the file as WARNING records as well. Child processes keep
    their inherited warning hook disabled to avoid stale or unsynchronized
    writes to the parent's file sink.

    Args:
        log_path: Destination file path (parent dirs created as needed).

    Returns:
        The destination path.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    global _file_handler
    old_handler = _file_handler

    # Build + verify the new sink before touching the old one, so a failure
    # (bad path, permissions) leaves the previous handler intact.
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    try:
        handler.setLevel(_get_log_level_from_env())
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        _write_session_header(handler)
    except Exception:
        handler.close()
        raise

    if old_handler is not None:
        _detach_from_all(old_handler)
        _detach_warning_handler(old_handler)
        old_handler.close()

    _file_handler = handler
    _warning_logger.setLevel(handler.level)
    if handler not in _warning_logger.handlers:
        _warning_logger.addHandler(handler)
    _install_warning_capture()
    for lg in _all_framework_loggers():
        if handler not in lg.handlers:
            lg.addHandler(handler)
    return log_path


def remove_file_handler() -> None:
    """Detach and close the shared file handler, if any.

    The warning hook is restored only when it is still owned by this module;
    an external replacement made while the sink was active is left intact.
    """
    global _file_handler
    _restore_warning_capture()
    if _file_handler is None:
        return
    handler = _file_handler
    _detach_from_all(handler)
    _detach_warning_handler(handler)
    handler.close()
    _file_handler = None
    _warning_logger.setLevel(logging.NOTSET)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with a unified configuration.

    Each logger carries a :class:`rich.logging.RichHandler` for console
    output (``propagate=False`` avoids duplicate messages) and, when a file
    sink is registered via :func:`add_file_handler`, that handler too.

    Args:
        name: Logger name, typically ``__name__`` for module-level logging.

    Returns:
        A configured ``logging.Logger`` instance.

    Example:
        >>> from utils.core import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        [14:30:25][INFO][my_module] This is an info message
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]

    # Create new logger
    logger = logging.getLogger(name)
    logger.setLevel(_get_log_level_from_env())
    logger.propagate = False

    # Console output via rich (SwanLab handles training metrics separately).
    if not logger.handlers:
        console_handler = RichHandler()
        console_handler.setLevel(_get_log_level_from_env())
        logger.addHandler(console_handler)

    # Pick up the shared file sink when it has been registered already.
    if _file_handler is not None and _file_handler not in logger.handlers:
        logger.addHandler(_file_handler)

    # Cache the logger
    _loggers[name] = logger
    return logger


def set_log_level(level: int) -> None:
    """Set the global log level programmatically.

    Updates every framework logger and handler. The shared file sink is
    updated directly so a stale ``_loggers`` cache (or orphan loggers living
    only in the global registry) cannot keep an old level.

    Args:
        level: A logging level constant (e.g. ``logging.DEBUG``,
               ``logging.INFO``).

    Example:
        >>> from utils.core import get_logger, set_log_level
        >>> import logging
        >>> set_log_level(logging.DEBUG)
        >>> logger = get_logger(__name__)
        >>> logger.debug("This will now be visible")
    """
    os.environ["LOG_LEVEL"] = logging.getLevelName(level)
    if _file_handler is not None:
        _file_handler.setLevel(level)
        _warning_logger.setLevel(level)
    for logger in _all_framework_loggers():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def reset_loggers() -> None:
    """Reset all cached loggers.

    Detaches the shared file handler before clearing the cache, so it is
    removed from every logger while the cache is still populated.

    Example:
        >>> from utils.core import reset_loggers
        >>> reset_loggers()
    """
    remove_file_handler()
    _loggers.clear()
    # Clear root logger handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


__all__ = [
    "add_file_handler",
    "get_logger",
    "remove_file_handler",
    "reset_loggers",
    "set_log_level",
]
