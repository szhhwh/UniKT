"""Tests for the schema reflection helper's docstring-help extraction.

Covers the Google-style fallback parser used when ``docstring_parser`` is
absent (conda / custom-interpreter environments without the package), its
parity with the real parser on every reflected config class, the
degraded-mode dispatch and signal, and the module's import side-effect
hygiene.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

if TYPE_CHECKING:
    from web.backend.services.schema_extractor import SchemaExtractor

import web.backend.services._schema_helper as helper

_REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _cls_with_doc(doc: str) -> type:
    @dataclass
    class WithDoc:
        x: int = 0

    WithDoc.__doc__ = doc
    return WithDoc


class TestFallbackParser:
    def test_single_line_entries(self) -> None:
        doc = "Summary.\n\nArgs:\n    a: help a.\n    b: help b.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "help a.", "b": "help b."}

    def test_multiline_description_joined_with_newlines(self) -> None:
        doc = "S.\n\nArgs:\n    a: first line\n        second line\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "first line\nsecond line"}

    def test_typed_entry_form(self) -> None:
        doc = "S.\n\nArgs:\n    a (int): typed help.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "typed help."}

    def test_typed_entry_nested_parens(self) -> None:
        # Parenthesis nesting in the type annotation must not break the
        # entry match (docstring_parser parses these fine).
        doc = "S.\n\nArgs:\n    bins (tuple(int, int)): bin edges.\n"
        assert helper._fallback_docstring_helps(doc) == {"bins": "bin edges."}
        deeper = "S.\n\nArgs:\n    m (Dict(str, List(int))): the mapping help.\n"
        assert helper._fallback_docstring_helps(deeper) == {"m": "the mapping help."}

    def test_section_aliases(self) -> None:
        # docstring_parser accepts these Google-style aliases; the fallback
        # must too or the whole section silently yields {}.
        for alias in ("Parameters", "Arguments", "Params", "Attributes"):
            doc = f"S.\n\n{alias}:\n    a: help a.\n"
            assert helper._fallback_docstring_helps(doc) == {"a": "help a."}, alias

    def test_trailing_note_section_not_an_entry(self) -> None:
        # A "Note:" paragraph indented like entries (HGKTConfig style) must
        # not register a bogus "Note" param.
        doc = (
            "S.\n\nArgs:\n    a: help a.\n\n"
            "    Note: no scheduler here,\n    unlike elsewhere.\n"
        )
        assert helper._fallback_docstring_helps(doc) == {"a": "help a."}

    def test_section_resumes_after_mid_note(self) -> None:
        # Entries documented after a Note:-style paragraph keep their helps
        # (docstring_parser also keeps them).
        doc = "S.\n\nArgs:\n    a: aa\n\n    Note: mid note\n\n    b: bb"
        assert helper._fallback_docstring_helps(doc) == {"a": "aa", "b": "bb"}

    def test_deeper_entries_survive_shallower_note(self) -> None:
        # Entries indented deeper than the Note: line must not be swallowed
        # as note text — a blank line ends the note paragraph.
        doc = "S.\n\nArgs:\n        a: aa\n\n    Note: see the paper.\n\n        b: bb"
        helps = helper._fallback_docstring_helps(doc)
        assert helps.get("b") == "bb"

    def test_deeper_entries_survive_adjacent_note(self) -> None:
        # Same, without a blank line after the note paragraph — entry-like
        # deeper lines end the note instead of being swallowed as its text.
        doc = "S.\n\nArgs:\n        a: aa\n\n    Note: see the paper.\n        b: bb\n        c: cc\n"
        helps = helper._fallback_docstring_helps(doc)
        assert helps.get("b") == "bb"
        assert helps.get("c") == "cc"

    def test_empty_inline_description_with_long_text(self) -> None:
        # No boundary newline when the entry line carries no inline text;
        # docstring_parser returns the long text alone.
        doc = "S.\n\nArgs:\n    a:\n        long only\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "long only"}

    def test_tab_indented_continuation_joins_description(self) -> None:
        # Tabs count their full width so a tab-indented continuation line is
        # consumed as description text, not dropped.
        doc = "S.\n\nArgs:\n    a: short\n\tlong tab line\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "short\nlong tab line"}

    def test_note_example_line_does_not_clobber_real_entry(self) -> None:
        # A param-like line inside a Note: example ("lr: 0.001 works best")
        # terminates the note and looks like an entry — first entry wins, so
        # the real lr help survives (docstring_parser keeps it too).
        doc = (
            "S.\n\nArgs:\n    lr: Learning rate.\n\n"
            "    Note: from the paper's grid:\n        lr: 0.001 works best\n"
        )
        helps = helper._fallback_docstring_helps(doc)
        assert helps.get("lr") == "Learning rate."

    def test_multiparagraph_entry_blanking_rules(self) -> None:
        # docstring_parser joins the entry line (short) and the deeper lines
        # (long) with a single newline; the leading blank is dropped and
        # inner blanks are kept.
        one_blank = "S.\n\nArgs:\n    a: p1.\n\n        p2.\n"
        assert helper._fallback_docstring_helps(one_blank) == {"a": "p1.\np2."}
        two_blanks = "S.\n\nArgs:\n    a: p1.\n\n        p2.\n\n        p3.\n"
        assert helper._fallback_docstring_helps(two_blanks) == {"a": "p1.\np2.\n\np3."}
        no_leading_blank = "S.\n\nArgs:\n    a: p1.\n        p2.\n\n        p3.\n"
        assert helper._fallback_docstring_helps(no_leading_blank) == {
            "a": "p1.\np2.\n\np3."
        }

    def test_deeper_entry_like_line_after_blank_joins_description(self) -> None:
        # Not a new entry: it would dict-overwrite and attach the WRONG help.
        doc = "S.\n\nArgs:\n    a: real help.\n\n        clarification: more\n"
        assert helper._fallback_docstring_helps(doc) == {
            "a": "real help.\nclarification: more"
        }

    def test_empty_and_missing_docstring(self) -> None:
        assert helper._fallback_docstring_helps("") == {}
        assert helper._fallback_docstring_helps("No sections at all.") == {}

    def test_non_entry_lines_in_section_skipped(self) -> None:
        doc = "S.\n\nArgs:\n    free text line\n    a: help a.\n"
        assert helper._fallback_docstring_helps(doc) == {"a": "help a."}


class TestDegradedMode:
    def test_missing_parser_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(helper, "_parse_docstring", None)
        cls = _cls_with_doc("S.\n\nArgs:\n    x: help x.\n")
        assert helper._parse_docstring_helps(cls) == {"x": "help x."}

    def test_installed_parser_is_consulted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A distinctive stub value proves dispatch reaches the installed
        # parser — the fallback alone would return the docstring's own text.
        class _StubParam:
            arg_name = "x"
            description = "stub value"

        class _StubDoc:
            params: ClassVar = [_StubParam()]

        monkeypatch.setattr(helper, "_parse_docstring", lambda doc: _StubDoc())
        cls = _cls_with_doc("S.\n\nArgs:\n    x: help x.\n")
        assert helper._parse_docstring_helps(cls) == {"x": "stub value"}

    def test_degraded_marker_emitted_on_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The marker is a stderr sentinel so stdout stays a pure envelope
        # channel; schema_extractor checks it once after the returncode gate.
        monkeypatch.setattr(helper, "_parse_docstring", None)
        helper._emit_degraded_marker()
        assert helper.DEGRADED_MARKER in capsys.readouterr().err

    def test_no_marker_when_parser_installed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        if helper._parse_docstring is None:
            pytest.skip("docstring_parser not installed in this environment")
        helper._emit_degraded_marker()
        assert capsys.readouterr().err == ""


class TestImportHygiene:
    def test_import_does_not_prepend_cwd_to_sys_path(self) -> None:
        # Module-scope sys.path.insert would leak into any importing process
        # (pytest); the insert must live in the __main__ block only.
        code = (
            "import sys; "
            f"sys.path.insert(0, {_REPO_ROOT!r}); "
            "import web.backend.services._schema_helper; "
            "print(sys.path[0])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/tmp",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() != "."


class TestParserParity:
    def test_fallback_matches_docstring_parser_on_all_config_classes(
        self, registry_snapshot: None
    ) -> None:
        """Every reflected config class (framework + all registered models)
        must expose identical per-field helps under both parsers."""
        if helper._parse_docstring is None:
            pytest.skip("docstring_parser not installed in this environment")
        try:
            import torch  # noqa: F401  trainer modules import it at top level
        except ImportError:
            # Torch-less-but-parser-having environments (web, docs) cannot
            # import any trainer module; parity over models needs the stack.
            pytest.skip("training stack unavailable: torch not installed")
        import model  # noqa: F401  triggers @register_model_config discovery
        from utils.core import MODEL_CONFIGS, get_supported_models

        model_classes = []
        for name in get_supported_models():
            # A trainer without a config class makes registry.get raise
            # KeyError — production emits an error envelope and the extractor
            # drops the model; parity just skips it. Per-model suppression so
            # one unimportable module cannot skip the other 70+.
            with contextlib.suppress(ImportError, KeyError):
                model_classes.append(MODEL_CONFIGS.get(name))
        from utils import config as cfg

        assert model_classes  # empty means discovery itself failed

        classes = [
            getattr(cfg, name)
            for name in (
                "CompileConfig",
                "DownloadConfig",
                "EarlyStoppingConfig",
                "GeneralConfig",
                "ProcessConfig",
                "RunDataConfig",
            )
        ] + model_classes
        assert len(classes) == 6 + len(model_classes)

        for cls in classes:
            reference = helper._parse_docstring_helps(cls)
            fallback = helper._fallback_docstring_helps(cls.__doc__ or "")
            for f in fields(cls):
                assert fallback.get(f.name) == reference.get(f.name), (
                    cls.__name__,
                    f.name,
                )
            # Ghost keys beyond real fields must still come from the reference
            # parse (the fallback never invents entries docstring_parser lacks).
            assert set(fallback) <= set(reference) | {f.name for f in fields(cls)}, (
                cls.__name__
            )


class TestExtractorRobustness:
    """schema_extractor must survive hostile helper stdout and crashes.

    Its modules import with web/backend on sys.path (their own top-level
    ``from config import ...`` layout), hence the path-shimmed fixture.
    """

    @pytest.fixture
    def extractor(self) -> tuple[ModuleType, SchemaExtractor]:
        backend_dir = str(
            Path(__file__).resolve().parent.parent.parent / "web" / "backend"
        )
        saved = list(sys.path)
        sys.path.insert(0, backend_dir)
        try:
            import services.schema_extractor as mod

            return mod, mod.SchemaExtractor(env_manager=None)
        finally:
            sys.path[:] = saved

    @staticmethod
    def _fake_run_result(
        stdout: str, returncode: int = 0, stderr: str = ""
    ) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr
        )

    def test_stdout_noise_without_type_key_is_ignored(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A dependency printing bare JSON (dict without "type", or a scalar)
        # during the helper's imports must not raise out of _extract.
        sx = extractor[1]
        noise = '{"step": 1}\n123\nnull\n'
        good = (
            '{"type": "models", "data": ["M"]}\n'
            '{"type": "schema", "model": "M", "data": []}\n'
        )
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: self._fake_run_result(noise + good)
        )
        sx.list_models()
        assert sx._models == ["M"]
        assert sx._loaded is True

    def test_extract_crash_does_not_wedge_loading(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If _extract raises anyway, _loading must reset so later callers do
        # not block forever on cond.wait().
        sx = extractor[1]
        calls = []

        def boom(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            raise RuntimeError("boom")

        monkeypatch.setattr(sx, "_extract", boom)
        sx.list_models()  # must return, not hang
        sx.list_models()  # and stay responsive
        assert len(calls) == 2

    def test_preprocess_helper_crash_returns_none(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Exit != 0 is a failure even when the degraded marker was printed
        # to stderr before the crash; get_preprocess_schema surfaces KeyError.
        sx = extractor[1]
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: self._fake_run_result(
                "", returncode=1, stderr=helper.DEGRADED_MARKER + "\n"
            ),
        )
        with pytest.raises(KeyError):
            sx.get_preprocess_schema("process")

    def test_colliding_noise_envelope_before_real_recovers(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The helper's imports run before it prints anything, so a colliding
        # dependency line realistically lands BEFORE the real envelope;
        # last-wins keeps the real one (matches main).
        sx = extractor[1]
        fake = '{"type": "models", "data": ["FAKE"]}\n'
        real = '{"type": "models", "data": ["M"]}\n'
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: self._fake_run_result(fake + real),
        )
        assert sx.list_models() == ["M"]

    def test_all_models_errored_degrades_once_with_log(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Every model in an error envelope (e.g. a framework default turned
        # non-JSON-serializable): the empty list is cached with a loud log
        # line instead of re-running the subprocess per request.
        sx = extractor[1]
        out = (
            '{"type": "models", "data": ["A", "B"]}\n'
            '{"type": "error", "model": "A"}\n'
            '{"type": "error", "model": "B"}\n'
        )
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: self._fake_run_result(out)
        )
        with caplog.at_level(logging.ERROR, logger=extractor[0].__name__):
            assert sx.list_models() == []
        assert sx._loaded is True  # degraded once and cached, no retry storm
        assert any("every model" in r.message for r in caplog.records)

    def test_framework_error_envelope_keeps_model_list(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A framework reflection failure emits one error envelope keyed on
        # None; the model list (printed before it) must still be served.
        sx = extractor[1]
        out = (
            '{"type": "models", "data": ["A"]}\n'
            '{"type": "error", "model": null, "error": "boom"}\n'
        )
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: self._fake_run_result(out)
        )
        assert sx.list_models() == ["A"]

    def test_waiter_does_not_serially_retry_failed_extraction(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A waiter whose extraction failed gets None (KeyError) instead of
        # promoting to its own subprocess run — N waiters must not mean N
        # sequential 60s runs.
        sx = extractor[1]
        started = threading.Event()
        first_done = threading.Event()
        calls = []

        def failing_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
            calls.append(1)
            started.set()
            first_done.wait(timeout=10)
            return self._fake_run_result("", returncode=1)

        monkeypatch.setattr("subprocess.run", failing_run)

        results = []

        def waiter() -> None:
            # Starts after the extractor is mid-run, so it waits.
            started.wait(timeout=10)
            try:
                sx.get_preprocess_schema("process")
                results.append("ok")
            except KeyError:
                results.append("keyerror")

        def first() -> None:
            try:
                sx.get_preprocess_schema("process")
                results.append("ok")
            except KeyError:
                results.append("keyerror")

        t1 = threading.Thread(target=first)
        t1.start()
        assert started.wait(timeout=10)
        t2 = threading.Thread(target=waiter)
        t2.start()
        time.sleep(0.2)  # let t2 reach the cond.wait
        first_done.set()
        t1.join(timeout=10)
        t2.join(timeout=10)
        assert len(calls) == 1  # the waiter did not re-run the subprocess
        assert results.count("keyerror") == 2

    def test_degraded_marker_logs_warning(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sx = extractor[1]
        out = '{"type": "preprocess_schema", "action": "process", "data": []}\n'
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: self._fake_run_result(
                out, stderr=helper.DEGRADED_MARKER + "\n"
            ),
        )
        with caplog.at_level(logging.WARNING, logger=extractor[0].__name__):
            groups = sx.get_preprocess_schema("process")
        assert groups == []
        assert any("docstring_parser" in r.message for r in caplog.records)

    def test_preprocess_cache_not_poisoned_by_reset_race(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A reset_cache() that lands mid-extraction must invalidate the
        # in-flight result: the stale schema is not committed.
        sx = extractor[1]
        started = threading.Event()
        release = threading.Event()
        good = '{"type": "preprocess_schema", "action": "process", "data": []}\n'

        def slow_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
            started.set()
            release.wait(timeout=10)
            return self._fake_run_result(good)

        monkeypatch.setattr("subprocess.run", slow_run)
        outcome: dict[str, str] = {}

        def extract() -> None:
            try:
                sx.get_preprocess_schema("process")
                outcome["result"] = "ok"
            except KeyError:
                outcome["result"] = "miss"

        t = threading.Thread(target=extract)
        t.start()
        assert started.wait(timeout=10)
        sx.reset_cache()  # invalidate while the subprocess is "running"
        release.set()
        t.join(timeout=10)
        # The raced result is neither cached NOR served — the leader sees the
        # same miss its waiters would, and a fresh request re-extracts.
        assert sx._preprocess_schemas.get("process") is None
        assert outcome["result"] == "miss"

    def test_preprocess_single_flight(
        self,
        extractor: tuple[ModuleType, SchemaExtractor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Concurrent misses for one action share a single subprocess.
        sx = extractor[1]
        calls = []
        good = '{"type": "preprocess_schema", "action": "process", "data": []}\n'

        def counting_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
            calls.append(1)
            return self._fake_run_result(good)

        monkeypatch.setattr("subprocess.run", counting_run)
        threads = [
            threading.Thread(target=lambda: sx.get_preprocess_schema("process"))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert sx._preprocess_schemas.get("process") == []
        assert len(calls) == 1
