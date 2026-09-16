"""Tests for the unified logger: cache semantics, file sink, level handling."""

import logging
import os
import sys
import warnings
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import IO, Any, cast

import pytest
from rich.logging import RichHandler

from utils.core.logger import (
    add_file_handler,
    get_logger,
    remove_file_handler,
    reset_loggers,
    set_log_level,
)


def _fresh_name(isolated_loggers: ModuleType, key: str) -> str:
    """A unique logger name never used by another test (the global logging
    registry returns the same object per name, so reuse would carry residue)."""
    return f"utest.logger.{key}"


def _write_warning_to_stream(
    message: str | Warning,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: IO[str] | None = None,
    line: str | None = None,
) -> None:
    """Provide a native-style warning hook for stderr assertions."""
    stream = file or sys.stderr
    stream.write(warnings.formatwarning(message, category, filename, lineno, line))


# --- get_logger ---


class TestGetLogger:
    def test_same_name_returns_cached_instance(
        self, isolated_loggers: ModuleType
    ) -> None:
        name = _fresh_name(isolated_loggers, "cache")
        assert get_logger(name) is get_logger(name)

    def test_propagate_false_and_single_rich_handler(
        self, isolated_loggers: ModuleType
    ) -> None:
        lg = get_logger(_fresh_name(isolated_loggers, "handlers"))
        assert lg.propagate is False
        rich_handlers = [h for h in lg.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) == 1

    def test_second_call_adds_no_extra_handlers(
        self, isolated_loggers: ModuleType
    ) -> None:
        name = _fresh_name(isolated_loggers, "twice")
        first = get_logger(name)
        n = len(first.handlers)
        assert len(get_logger(name).handlers) == n

    def test_file_sink_attached_to_new_logger(
        self, isolated_loggers: ModuleType, tmp_path: Path
    ) -> None:
        add_file_handler(tmp_path / "run.log")
        try:
            lg = get_logger(_fresh_name(isolated_loggers, "after_sink"))
            assert isolated_loggers._file_handler in lg.handlers
        finally:
            remove_file_handler()


# --- add/remove file handler ---


class TestFileHandler:
    def test_creates_parent_dirs_and_returns_path(
        self, isolated_loggers: ModuleType, tmp_path: Path
    ) -> None:
        target = tmp_path / "nested" / "dirs" / "run.log"
        returned = add_file_handler(target)
        try:
            assert returned == target
            assert target.exists()
        finally:
            remove_file_handler()

    def test_session_header_contains_timestamp_and_pid(
        self, isolated_loggers: ModuleType, tmp_path: Path
    ) -> None:
        target = tmp_path / "run.log"
        add_file_handler(target)
        try:
            content = target.read_text(encoding="utf-8")
            assert "session start" in content
            assert f"pid={os.getpid()}" in content
            assert "=" * 70 in content
        finally:
            remove_file_handler()

    def test_retroactive_attach_to_existing_logger(
        self, isolated_loggers: ModuleType, tmp_path: Path
    ) -> None:
        lg = get_logger(_fresh_name(isolated_loggers, "retro"))
        target = tmp_path / "run.log"
        add_file_handler(target)
        try:
            assert any(
                isinstance(h, logging.FileHandler) and h.baseFilename == str(target)
                for h in lg.handlers
            )
        finally:
            remove_file_handler()

    def test_remove_without_handler_is_noop(self, isolated_loggers: ModuleType) -> None:
        remove_file_handler()  # must not raise
        remove_file_handler()

    def test_re_attach_swaps_and_closes_old_sink(
        self, isolated_loggers: ModuleType, tmp_path: Path
    ) -> None:
        first = tmp_path / "first.log"
        second = tmp_path / "second.log"
        lg = get_logger(_fresh_name(isolated_loggers, "swap"))
        add_file_handler(first)
        lg.warning("to first")
        add_file_handler(second)
        try:
            lg.warning("to second")
            first_lines = first.read_text(encoding="utf-8")
            second_lines = second.read_text(encoding="utf-8")
            assert "to first" in first_lines
            assert "to second" not in first_lines  # old sink closed, no appends
            assert "to second" in second_lines
        finally:
            remove_file_handler()

    def test_failed_attach_leaves_previous_sink(
        self, isolated_loggers: ModuleType, tmp_path: Path
    ) -> None:
        good = tmp_path / "good.log"
        add_file_handler(good)
        # A directory in place of the log file makes FileHandler construction fail.
        (tmp_path / "not_a_file.log").mkdir()
        # OSError message varies by platform; only the type is asserted.
        with pytest.raises(OSError):
            add_file_handler(tmp_path / "not_a_file.log")
        try:
            from utils.core import logger as logger_module

            assert cast(
                logging.FileHandler, logger_module._file_handler
            ).baseFilename == str(good)
        finally:
            remove_file_handler()


# --- level handling ---


class TestLevelHandling:
    def test_env_debug_level(
        self, isolated_loggers: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        lg = get_logger(_fresh_name(isolated_loggers, "debug"))
        assert lg.level == logging.DEBUG

    def test_env_unknown_falls_back_to_info(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "notalevel")
        lg = get_logger(_fresh_name(isolated_loggers, "fallback"))
        assert lg.level == logging.INFO

    def test_set_log_level_updates_cached_and_future(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        name = _fresh_name(isolated_loggers, "setlevel")
        existing = get_logger(name)
        set_log_level(logging.ERROR)
        try:
            assert existing.level == logging.ERROR
            future = get_logger(_fresh_name(isolated_loggers, "setlevel2"))
            assert future.level == logging.ERROR
            assert os.environ["LOG_LEVEL"] == "ERROR"
        finally:
            set_log_level(logging.INFO)

    def test_set_log_level_updates_file_sink(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from utils.core import logger as logger_module

        add_file_handler(tmp_path / "run.log")
        try:
            set_log_level(logging.DEBUG)
            assert (
                cast(logging.FileHandler, logger_module._file_handler).level
                == logging.DEBUG
            )
        finally:
            remove_file_handler()
            set_log_level(logging.INFO)


# --- warnings capture ---


class TestWarningsCapture:
    def test_warning_written_to_file_sink(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Bypass pytest's warning recorder so this asserts the real stderr path.
        monkeypatch.setattr(warnings, "showwarning", _write_warning_to_stream)
        target = tmp_path / "run.log"
        add_file_handler(target)
        try:
            with warnings.catch_warnings():
                # Bypass the default once-per-location warning filter.
                warnings.simplefilter("always")
                warnings.warn("third-party style warning", RuntimeWarning)
            content = target.read_text(encoding="utf-8")
            captured = capsys.readouterr()
            assert "third-party style warning" in content
            assert "WARNING" in content
            assert "third-party style warning" in captured.err
        finally:
            remove_file_handler()

    def test_warning_at_error_level_still_reaches_stderr(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        monkeypatch.setattr(warnings, "showwarning", _write_warning_to_stream)
        target = tmp_path / "run.log"
        add_file_handler(target)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("visible error-level warning", RuntimeWarning)
            content = target.read_text(encoding="utf-8")
            captured = capsys.readouterr()
            assert "visible error-level warning" not in content
            assert "visible error-level warning" in captured.err
        finally:
            remove_file_handler()

    def test_remove_restores_native_warnings_path(
        self, isolated_loggers: ModuleType, clean_log_level_env: None, tmp_path: Path
    ) -> None:
        native_showwarning = warnings.showwarning
        add_file_handler(tmp_path / "run.log")
        assert warnings.showwarning is not native_showwarning
        remove_file_handler()
        assert warnings.showwarning is native_showwarning

    def test_remove_preserves_preexisting_warning_capture(
        self, isolated_loggers: ModuleType, clean_log_level_env: None, tmp_path: Path
    ) -> None:
        original_showwarning = warnings.showwarning
        original_saved_hook = getattr(logging, "_warnings_showwarning", None)
        logging.captureWarnings(True)
        preexisting_hook = warnings.showwarning
        try:
            add_file_handler(tmp_path / "run.log")
            remove_file_handler()
            assert warnings.showwarning is preexisting_hook
        finally:
            remove_file_handler()
            warnings.showwarning = original_showwarning
            setattr(logging, "_warnings_showwarning", original_saved_hook)

    def test_remove_leaves_external_warning_replacement(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        add_file_handler(tmp_path / "run.log")

        def replacement(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(warnings, "showwarning", replacement)
        remove_file_handler()
        assert warnings.showwarning is replacement

    def test_warning_sink_does_not_mutate_stdlib_warning_logger(
        self, isolated_loggers: ModuleType, clean_log_level_env: None, tmp_path: Path
    ) -> None:
        py_warnings = logging.getLogger("py.warnings")
        original_level = py_warnings.level
        original_propagate = py_warnings.propagate
        external_handler = logging.StreamHandler()
        external_handler.setLevel(logging.ERROR)
        py_warnings.addHandler(external_handler)
        try:
            add_file_handler(tmp_path / "run.log")
            set_log_level(logging.DEBUG)
            assert py_warnings.level == original_level
            assert py_warnings.propagate is original_propagate
            assert external_handler.level == logging.ERROR
        finally:
            remove_file_handler()
            py_warnings.removeHandler(external_handler)
            external_handler.close()
            set_log_level(logging.INFO)

    def test_warning_sink_does_not_propagate_to_root(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(warnings, "showwarning", _write_warning_to_stream)
        root_logger = logging.getLogger()
        root_stream = StringIO()
        root_handler = logging.StreamHandler(root_stream)
        root_logger.addHandler(root_handler)
        try:
            add_file_handler(tmp_path / "run.log")
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("no-root-warning", UserWarning)
            assert "no-root-warning" not in root_stream.getvalue()
        finally:
            remove_file_handler()
            root_logger.removeHandler(root_handler)
            root_handler.close()

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
    def test_forked_worker_does_not_write_to_parent_sink(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(warnings, "showwarning", _write_warning_to_stream)
        target = tmp_path / "run.log"
        add_file_handler(target)
        pid = os.fork()
        if pid == 0:
            try:
                with open(os.devnull, "w") as devnull:
                    os.dup2(devnull.fileno(), 2)
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn("worker-only-warning", UserWarning)
            finally:
                os._exit(0)
        try:
            _, wait_status = os.waitpid(pid, 0)
            assert os.WIFEXITED(wait_status)
            assert "worker-only-warning" not in target.read_text(encoding="utf-8")
        finally:
            remove_file_handler()

    def test_re_attach_moves_warnings_sink(
        self,
        isolated_loggers: ModuleType,
        clean_log_level_env: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            warnings, "showwarning", getattr(warnings, "_showwarning_orig")
        )
        first = tmp_path / "first.log"
        second = tmp_path / "second.log"
        add_file_handler(first)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("warning to first", UserWarning)
        add_file_handler(second)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("warning to second", UserWarning)
            first_content = first.read_text(encoding="utf-8")
            second_content = second.read_text(encoding="utf-8")
            assert "warning to first" in first_content
            assert "warning to second" not in first_content
            assert "warning to second" in second_content
        finally:
            remove_file_handler()


# --- reset ---


class TestResetLoggers:
    def test_reset_clears_cache_and_detaches_sink(
        self, isolated_loggers: ModuleType, tmp_path: Path
    ) -> None:
        from utils.core import logger as logger_module

        name = _fresh_name(isolated_loggers, "reset")
        lg = get_logger(name)
        add_file_handler(tmp_path / "run.log")
        assert logger_module._file_handler is not None
        reset_loggers()
        assert logger_module._file_handler is None
        assert name not in logger_module._loggers
        assert lg not in logger_module._loggers.values()
