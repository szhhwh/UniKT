"""Tests for UniversalRegistry: decorator registration, static index, lazy get."""

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from utils.core.registry import (
    MODEL_CONFIGS,
    UniversalRegistry,
    register_model_config,
    register_trainer,
)


@pytest.fixture
def fresh_registry() -> Iterator[UniversalRegistry]:
    """A private registry appended to the roll-call; removed on teardown."""
    reg = UniversalRegistry("utest_reg", decorator_name="register_utest")
    yield reg
    UniversalRegistry._all_registries.remove(reg)


# --- register ---


class TestRegister:
    def test_same_class_reregister_is_idempotent(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        @fresh_registry.register("Dup")
        class Dup: ...

        # Re-import style: applying the same decorator to the same class again
        fresh_registry.register("Dup")(Dup)
        assert fresh_registry._registry["Dup"] is Dup

    def test_different_class_same_name_raises(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        @fresh_registry.register("Clash")
        class First: ...

        with pytest.raises(KeyError, match="already registered"):
            fresh_registry.register("Clash")(type("Second", (), {}))

    def test_name_defaults_to_class_name(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        @fresh_registry.register()
        class DefaultName: ...

        assert fresh_registry._registry["DefaultName"] is DefaultName

    def test_register_pops_stale_index_entry(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        fresh_registry.index("Soon", "some.module")
        fresh_registry.register("Soon")(type("Soon", (), {}))
        assert "Soon" not in fresh_registry._index
        assert "Soon" in fresh_registry._registry


# --- index ---


class TestIndex:
    def test_same_path_twice_is_noop(self, fresh_registry: UniversalRegistry) -> None:
        fresh_registry.index("Entry", "pkg.mod")
        fresh_registry.index("Entry", "pkg.mod")
        assert fresh_registry._index["Entry"] == "pkg.mod"

    def test_different_path_same_name_raises(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        fresh_registry.index("Entry", "pkg.mod_a")
        with pytest.raises(KeyError, match="indexed twice"):
            fresh_registry.index("Entry", "pkg.mod_b")

    def test_index_only_entry_visible_in_protocol(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        fresh_registry.index("Only", "pkg.mod")
        assert "Only" in fresh_registry
        assert fresh_registry.keys() == ["Only"]
        assert len(fresh_registry) == 1
        assert list(fresh_registry) == ["Only"]
        assert bool(fresh_registry)


# --- get ---


class TestGet:
    def test_loaded_entry_returned_directly(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        cls = type("Loaded", (), {})
        fresh_registry.register("Loaded")(cls)
        assert fresh_registry.get("Loaded") is cls

    def test_indexed_entry_lazy_imports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry_snapshot: None
    ) -> None:
        from utils.core import TRAINERS

        pkg = tmp_path / "utest_lazy_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "mod.py").write_text(
            "from utils.core import register_trainer\n"
            "\n"
            "@register_trainer('LazyWidget')\n"
            "class LazyWidget:\n"
            "    marker = 'lazy'\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "utest_lazy_pkg", raising=False)
        monkeypatch.delitem(sys.modules, "utest_lazy_pkg.mod", raising=False)

        # Index + decorator must target the same (global) registry so the
        # lazy import actually lands where get() looks for it.
        TRAINERS.index("LazyWidget", "utest_lazy_pkg.mod")
        # get() returns a dynamically imported class of unknown shape.
        lazy_cls: Any = TRAINERS.get("LazyWidget")
        assert lazy_cls.marker == "lazy"
        assert TRAINERS._registry["LazyWidget"] is lazy_cls
        assert "LazyWidget" not in TRAINERS._index

    def test_unknown_name_raises_with_available(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        fresh_registry.register("Known")(type("Known", (), {}))
        with pytest.raises(KeyError, match=r"not found.*Known"):
            fresh_registry.get("Missing")

    def test_broken_module_path_propagates(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        fresh_registry.index("Broken", "utest_no_such_module_anywhere")
        with pytest.raises(ModuleNotFoundError):
            fresh_registry.get("Broken")


# --- clear ---


class TestClear:
    def test_clear_empties_both_tables(self, fresh_registry: UniversalRegistry) -> None:
        fresh_registry.register("R")(type("R", (), {}))
        fresh_registry.index("I", "pkg.mod")
        fresh_registry.clear()
        assert not fresh_registry
        assert fresh_registry._registry == {}
        assert fresh_registry._index == {}


# --- dunder protocol ---


class TestDunderProtocol:
    def test_keys_registry_before_index_dedupe(
        self, fresh_registry: UniversalRegistry
    ) -> None:
        fresh_registry.register("Shared")(type("Shared", (), {}))
        fresh_registry.index("Shared", "pkg.mod")  # shadowed by registry
        fresh_registry.index("Extra", "pkg.mod")
        assert fresh_registry.keys() == ["Shared", "Extra"]

    def test_len_contains_iter_agree(self, fresh_registry: UniversalRegistry) -> None:
        fresh_registry.register("A")(type("A", (), {}))
        fresh_registry.index("B", "pkg.b")
        assert set(fresh_registry) == {"A", "B"}
        assert len(fresh_registry) == 2
        assert "A" in fresh_registry and "B" in fresh_registry
        assert "C" not in fresh_registry

    def test_repr_contains_name(self, fresh_registry: UniversalRegistry) -> None:
        assert "utest_reg" in repr(fresh_registry)


# --- global registries ---


class TestGlobalRegistries:
    def test_global_registries_declare_expected_decorator_names(self) -> None:
        from utils.core import registry as registry_module

        expected = {
            "TRAINERS": "register_trainer",
            "MODEL_CONFIGS": "register_model_config",
            "DATA_SOURCES": "register_data_source",
            "ANALYZERS": "register_analyzer",
            "METRIC_LOGGERS": None,  # import-time only, not discovered
            "EFFICIENCY_STAGES": "register_efficiency_stage",
            "METRICS": "register_metric",
            "CASE_SINKS": "register_case_sink",
            "CASE_SELECTORS": "register_case_selector",
            "CASE_VISUALIZERS": "register_case_visualizer",
        }
        for attr, decorator_name in expected.items():
            reg = getattr(registry_module, attr)
            assert isinstance(reg, UniversalRegistry)
            assert reg.decorator_name == decorator_name

    def test_all_registries_on_roll_call(self) -> None:
        names = {r._name for r in UniversalRegistry._all_registries}
        assert {
            "trainers",
            "model_configs",
            "data_sources",
            "analyzers",
            "metric_loggers",
            "efficiency_stages",
            "metrics",
            "case_sinks",
            "case_selectors",
            "case_visualizers",
        } <= names

    def test_register_model_config_applies_dataclass(
        self, registry_snapshot: None
    ) -> None:
        class RawConfig:  # deliberately not a dataclass before decoration
            epochs: int = 2

        decorated = register_model_config("UTestModel")(RawConfig)
        assert MODEL_CONFIGS._registry["UTestModel"] is decorated
        assert decorated().epochs == 2
        # is_dataclass is a TypeGuard and would narrow the class to the opaque
        # DataclassInstance protocol, so it must come after the attribute checks.
        import dataclasses

        assert dataclasses.is_dataclass(decorated)

    def test_register_trainer_wraps_trainers(self, registry_snapshot: None) -> None:
        @register_trainer("UTestTrainer")
        class UTestTrainer: ...

        from utils.core import TRAINERS

        assert TRAINERS.get("UTestTrainer") is UTestTrainer
