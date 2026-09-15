"""Tests for static discovery: AST scan of @register_<role>("literal") forms.

All fixtures are synthetic package trees under tmp_path — the real ``model/``
tree is never scanned, keeping the tests hermetic and torch-free.
"""

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from utils.core.discovery import (
    _decorator_map,
    _to_module_path,
    discover_registrations,
)
from utils.core.registry import UniversalRegistry


def _write_module(root: Path, rel_path: str, content: str) -> Path:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def pkg_root(tmp_path: Path) -> Path:
    return tmp_path / "utest_scan_pkg"


@pytest.fixture(autouse=True)
def _restore_trainers_index() -> Iterator[None]:
    """discover_registrations writes into the global TRAINERS index."""
    from utils.core import TRAINERS

    saved = dict(TRAINERS._index)
    yield
    TRAINERS._index.clear()
    TRAINERS._index.update(saved)


# --- discover_registrations ---


class TestDiscoverRegistrations:
    def test_bare_decorator_populates_index(
        self, pkg_root: Path, registry_snapshot: None
    ) -> None:
        from utils.core import TRAINERS

        _write_module(
            pkg_root,
            "trainer_a.py",
            "from utils.core import register_trainer\n"
            "\n"
            "@register_trainer('Alpha')\n"
            "class Alpha: ...\n",
        )
        discover_registrations(pkg_root, "utest_scan_pkg")
        assert TRAINERS._index["Alpha"] == "utest_scan_pkg.trainer_a"

    def test_nested_directory_dotted_path(
        self, pkg_root: Path, registry_snapshot: None
    ) -> None:
        from utils.core import TRAINERS

        _write_module(
            pkg_root,
            "sub/dir/mod_b.py",
            "from utils.core import register_trainer\n"
            "\n"
            "@register_trainer('Beta')\n"
            "class Beta: ...\n",
        )
        discover_registrations(pkg_root, "utest_scan_pkg")
        assert TRAINERS._index["Beta"] == "utest_scan_pkg.sub.dir.mod_b"

    def test_init_py_skipped(self, pkg_root: Path, registry_snapshot: None) -> None:
        from utils.core import TRAINERS

        _write_module(
            pkg_root,
            "__init__.py",
            "from utils.core import register_trainer\n"
            "@register_trainer('Gamma')\n"
            "class Gamma: ...\n",
        )
        discover_registrations(pkg_root, "utest_scan_pkg")
        assert "Gamma" not in TRAINERS._index

    def test_syntax_error_skipped_with_warning(
        self,
        pkg_root: Path,
        registry_snapshot: None,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import logging

        from utils.core import TRAINERS

        # Framework loggers are propagate=False, so caplog only sees them
        # once propagation is temporarily re-enabled for the discovery logger.
        monkeypatch.setattr(
            logging.getLogger("utils.core.discovery"), "propagate", True
        )

        _write_module(pkg_root, "broken.py", "def (:\n")
        _write_module(
            pkg_root,
            "good.py",
            "from utils.core import register_trainer\n"
            "@register_trainer('Delta')\n"
            "class Delta: ...\n",
        )
        with caplog.at_level("WARNING"):
            discover_registrations(pkg_root, "utest_scan_pkg")
        assert any("broken.py" in r.getMessage() for r in caplog.records)
        assert "Delta" in TRAINERS._index  # scan continues past the bad file

    def test_nonliteral_argument_ignored(
        self, pkg_root: Path, registry_snapshot: None
    ) -> None:
        from utils.core import TRAINERS

        _write_module(
            pkg_root,
            "var_name.py",
            "NAME = 'Epsilon'\n"
            "from utils.core import register_trainer\n"
            "@register_trainer(NAME)\n"
            "class Epsilon: ...\n",
        )
        discover_registrations(pkg_root, "utest_scan_pkg")
        assert "Epsilon" not in TRAINERS._index

    def test_attribute_call_decorator_ignored(
        self, pkg_root: Path, registry_snapshot: None
    ) -> None:
        from utils.core import TRAINERS

        _write_module(
            pkg_root,
            "attr_call.py",
            "import utils.core as uc\n@uc.register_trainer('Zeta')\nclass Zeta: ...\n",
        )
        discover_registrations(pkg_root, "utest_scan_pkg")
        assert "Zeta" not in TRAINERS._index

    def test_function_decorator_ignored(
        self, pkg_root: Path, registry_snapshot: None
    ) -> None:
        from utils.core import TRAINERS

        _write_module(
            pkg_root,
            "func_dec.py",
            "from utils.core import register_trainer\n"
            "@register_trainer('Eta')\n"
            "def eta(): ...\n",
        )
        discover_registrations(pkg_root, "utest_scan_pkg")
        assert "Eta" not in TRAINERS._index

    def test_undiscovered_registry_not_populated(
        self, pkg_root: Path, registry_snapshot: None
    ) -> None:
        from utils.core import METRIC_LOGGERS

        _write_module(
            pkg_root,
            "ml.py",
            "from utils.core import register_metric_logger\n"
            "@register_metric_logger('Theta')\n"
            "class Theta: ...\n",
        )
        discover_registrations(pkg_root, "utest_scan_pkg")
        # METRIC_LOGGERS has no decorator_name -> invisible to discovery
        assert "Theta" not in METRIC_LOGGERS._index

    def test_duplicate_name_from_two_modules_raises(
        self, pkg_root: Path, registry_snapshot: None
    ) -> None:
        _write_module(
            pkg_root,
            "first.py",
            "from utils.core import register_trainer\n"
            "@register_trainer('Iota')\n"
            "class Iota: ...\n",
        )
        _write_module(
            pkg_root,
            "second.py",
            "from utils.core import register_trainer\n"
            "@register_trainer('Iota')\n"
            "class Iota2: ...\n",
        )
        with pytest.raises(KeyError, match="indexed twice"):
            discover_registrations(pkg_root, "utest_scan_pkg")

    def test_round_trip_get_imports_class(
        self,
        pkg_root: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        registry_snapshot: None,
    ) -> None:
        from utils.core import TRAINERS

        _write_module(
            pkg_root,
            "round_trip.py",
            "from utils.core import register_trainer\n"
            "\n"
            "@register_trainer('Kappa')\n"
            "class Kappa:\n"
            "    marker = 'k'\n",
        )
        pkg_root.mkdir(parents=True, exist_ok=True)
        init = pkg_root / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "utest_scan_pkg", raising=False)
        monkeypatch.delitem(sys.modules, "utest_scan_pkg.round_trip", raising=False)

        discover_registrations(pkg_root, "utest_scan_pkg")
        # get() returns a dynamically imported class of unknown shape.
        kappa_cls: Any = TRAINERS.get("Kappa")
        assert kappa_cls.marker == "k"


# --- helpers ---


class TestHelpers:
    def test_decorator_map_covers_declared_registries(
        self, registry_snapshot: None
    ) -> None:
        UniversalRegistry("utest_map_reg", decorator_name="register_utest_map")
        mapping = _decorator_map()
        assert mapping["register_trainer"]._name == "trainers"
        assert mapping["register_utest_map"]._name == "utest_map_reg"
        assert all(name for name in mapping)

    def test_to_module_path_depths(self, tmp_path: Path) -> None:
        from pathlib import Path

        root = tmp_path / "pkg"
        assert _to_module_path(root / "top.py", root, "pkg") == "pkg.top"
        assert _to_module_path(root / "a" / "mid.py", root, "pkg") == "pkg.a.mid"
        assert (
            _to_module_path(root / "a" / "b" / "c.py", root, "utils.data_process")
            == "utils.data_process.a.b.c"
        )
        # Path() keeps the signature honest even for silly inputs
        assert isinstance(_to_module_path(Path(root) / "x.py", root, "p"), str)
