"""Tests for env loading: override semantics and module-anchored search."""

import os

import pytest

from utils.core.env import load_env


class TestLoadEnv:
    def test_existing_process_vars_take_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # override=False: a var already in the environment survives loading,
        # even if a .env somewhere defines it.
        monkeypatch.setenv("UTEST_ENV_PROBE", "process-value")
        load_env()
        load_env()  # idempotent on repeat calls
        assert os.environ["UTEST_ENV_PROBE"] == "process-value"

    def test_search_is_module_anchored_not_cwd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}
        monkeypatch.setattr(
            "utils.core.env.find_dotenv",
            lambda usecwd: calls.setdefault("usecwd", usecwd) or "",
        )
        monkeypatch.setattr(
            "utils.core.env.load_dotenv",
            lambda path, override: calls.update(path=path, override=override),
        )
        load_env()
        assert calls == {"usecwd": False, "path": "", "override": False}
