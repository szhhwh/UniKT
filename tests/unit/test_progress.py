"""Tests for the shared Rich progress bar factory."""

import sys

import pytest
from rich.progress import Progress

from utils.progress import create_progress, resolve_progress


class TestCreateProgress:
    def test_returns_progress_instance(self) -> None:
        prog = create_progress()
        assert isinstance(prog, Progress)

    def test_two_calls_independent(self) -> None:
        assert create_progress() is not create_progress()

    def test_add_task_and_advance(self) -> None:
        prog = create_progress()
        task_id = prog.add_task("working", total=4)
        prog.advance(task_id, 2)
        task = prog.tasks[task_id]
        assert task.completed == 4 // 2
        prog.stop()


class TestResolveProgress:
    def test_rich_always_enables(self) -> None:
        assert resolve_progress("rich") is True

    def test_none_always_disables(self) -> None:
        assert resolve_progress("none") is False

    def test_auto_follows_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert resolve_progress("auto") is True
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        assert resolve_progress("auto") is False

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown progress mode"):
            resolve_progress("bogus")
