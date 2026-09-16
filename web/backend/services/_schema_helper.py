"""Schema reflection helper — run as a subprocess via SchemaExtractor.

Two modes (argv[1]):
  - ``models`` (default): reflect every registered model's config + the fixed
    framework groups; emits ``models`` / ``schema`` / ``error`` envelopes.
  - ``preprocess <action>``: reflect the data_process.py download/process
    parameter dataclasses (with a UI-level field whitelist); emits one
    ``preprocess_schema`` envelope.

The reflection unit is a dataclass: ``reflect_group`` walks
``dataclasses.fields(cls)`` and maps each field to ``{type, default, help,
choices, ...}``, so model configs and preprocess configs share one path.
"""

import json
import re
import sys
import typing
from dataclasses import MISSING, Field, fields
from pathlib import Path

# Every pixi environment declares docstring-parser at workspace level; the
# remaining package-less audiences are wizard-configured
# conda/custom interpreters. Schema reflection must still run there — hence
# the fallback parser. Exception, not just ImportError: a corrupt install
# (SyntaxError) must degrade to the fallback too, and since schema_extractor
# imports this module for DEGRADED_MARKER, a narrow catch would let a broken
# package take the whole backend down at boot.
try:
    from docstring_parser import parse as _parse_docstring_impl

    _parse_docstring: typing.Callable[..., typing.Any] | None = _parse_docstring_impl
except Exception:
    _parse_docstring = None

_SECTION_RE = re.compile(r"^(Args|Arguments|Attributes|Parameters|Params):$")
# Other Google-style sections that may appear indented like an entry inside
# an Args/Attributes block (e.g. a trailing "Note:" paragraph); the block
# resumes after their paragraph instead of ending.
_ENTRY_STOP_RE = re.compile(
    r"^(Note|Notes|Returns|Raises|Examples?|Yields|Warnings?|See Also|References):"
)
# Typed entries are ``name (type): desc`` — the type group spans any
# parenthesis content without a colon (nesting depth is irrelevant because
# the colon terminates it).
_PARAM_RE = re.compile(r"^(\w+)(?:\s*\([^:]*\))?\s*:\s*(.*)$")


def _base_type(tp: typing.Any) -> typing.Any:
    # ``tp`` is a typing form: a plain type, a generic alias, or a string
    # under PEP 563 — no narrower public type covers all three.
    args = typing.get_args(tp)
    if not args:
        return tp
    if typing.get_origin(tp) is list:
        return args[0] if args else tp
    non_none = [a for a in args if a is not type(None)]
    if type(None) in args and len(non_none) == 1:
        return non_none[0]
    return tp


def _field_spec(
    f: Field[typing.Any], help_map: dict[str, str] | None = None
) -> dict[str, typing.Any]:
    ftype = f.type
    meta = f.metadata

    # Extract Literal choices from the annotation when metadata lacks them
    # (e.g. compile_mode: Literal["default", "reduce-overhead", ...]).
    choices = meta.get("choices")
    if choices is None:
        origin = typing.get_origin(ftype)
        if origin is typing.Literal:
            choices = list(typing.get_args(ftype))

    if typing.get_origin(ftype) is list:
        type_str = "list"
    elif ftype is bool:
        type_str = "bool"
    else:
        scalar = _base_type(ftype)
        type_str = {int: "int", float: "float", str: "str", bool: "bool"}.get(
            scalar, "str"
        )
    default = f.default_factory() if f.default_factory is not MISSING else f.default

    # Prefer metadata help; fall back to the class docstring Args: section.
    help_text = meta.get("help", "")
    if not help_text and help_map:
        help_text = help_map.get(f.name, "")

    return {
        "type": type_str,
        "default": default,
        "help": help_text,
        "required": False,
        "choices": choices,
        "short": meta.get("short"),
        "nargs": meta.get("nargs"),
        # Search-space spec (low/high/log/step/choices) for fields carrying
        # ``optuna`` metadata; None on non-searchable fields.
        "optuna": meta.get("optuna"),
    }


def _fallback_docstring_helps(doc: str) -> dict[str, str]:
    """Extract ``{name: description}`` from Google-style Args/Attributes sections.

    Covers the entry forms used in this repo: ``name: desc`` and
    ``name (type): desc`` (one nesting level of parentheses) with
    continuation lines, matching docstring_parser's Google-style output:
    the entry line is the short description, deeper-indented lines form the
    long description (leading blanks dropped, inner blank lines kept), and
    the two join with a single newline. NumPy/reST-style docstrings are NOT
    covered — install docstring-parser for those.
    """
    helps: dict[str, str] = {}
    lines = doc.splitlines()

    def indent(idx: int) -> int:
        # expandtabs so tab-indented continuation lines count their full width
        line = lines[idx].expandtabs()
        return len(line) - len(line.lstrip())

    i = 0
    while i < len(lines):
        if not _SECTION_RE.match(lines[i].strip()):
            i += 1
            continue
        i += 1
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            if not line[0].isspace():
                break  # section ended (section header or summary text)
            if _ENTRY_STOP_RE.match(stripped):
                # Skip the Note:/Returns:-style paragraph and resume the
                # section instead of dropping entries documented after it.
                # Only its immediately-indented non-entry lines belong to the
                # note; blank lines and entry-like lines end it so following
                # entries stay parseable.
                note_indent = indent(i)
                i += 1
                while i < len(lines):
                    s = lines[i].strip()
                    if s and indent(i) > note_indent and not _PARAM_RE.match(s):
                        i += 1
                    else:
                        break
                continue
            m = _PARAM_RE.match(stripped)
            if m is None:
                i += 1
                continue
            base_indent = indent(i)
            name = m.group(1)
            short = m.group(2).strip()
            long_lines: list[str] = []
            i += 1
            # Collect the raw continuation block (blank or deeper-indented
            # lines), then trim the blank runs at both ends — inner blanks
            # stay as paragraph separators, as docstring_parser does.
            while i < len(lines):
                s = lines[i].strip()
                if s and indent(i) <= base_indent:
                    break
                long_lines.append(s)
                i += 1
            while long_lines and not long_lines[0]:
                long_lines.pop(0)
            while long_lines and not long_lines[-1]:
                long_lines.pop()
            text = short
            if long_lines:
                # Empty inline descriptions carry no boundary newline —
                # docstring_parser returns the long text alone.
                long_text = "\n".join(long_lines)
                text = short + "\n" + long_text if short else long_text
            if text and name not in helps:
                # First entry wins: a param-like line inside a Note: example
                # ("lr: 0.001 works best") must not clobber the real entry.
                helps[name] = text
    return helps


def _parse_docstring_helps(cls: type) -> dict[str, str]:
    """Extract ``{field_name: help_text}`` from the class docstring Args: section.

    Delegates to ``docstring_parser`` (Google/NumPy/Sphinx, multi-line) when
    installed; otherwise falls back to a minimal Google-style parser so
    environments without the package still expose parameter helps.
    """
    doc = cls.__doc__ or ""
    if _parse_docstring is not None:
        return {
            p.arg_name: p.description
            for p in _parse_docstring(doc).params
            if p.description
        }
    return _fallback_docstring_helps(doc)


def reflect_group(
    group_name: str,
    node: str,
    cls: type,
    only: set[str] | None = None,
    skip: set[str] | None = None,
) -> dict[str, typing.Any]:
    """Reflect a dataclass into a ``{group_name, node, params}`` schema group.

    ``only``/``skip`` restrict which fields are exposed — a UI-level selection
    for the preprocess schema (which need not surface every RunDataConfig /
    GeneralConfig knob); the model schema leaves them unset to expose all.
    """
    help_map = _parse_docstring_helps(cls)
    params = {}
    for f in fields(cls):
        if only is not None and f.name not in only:
            continue
        if skip and f.name in skip:
            continue
        params[f.name] = _field_spec(f, help_map)
    return {"group_name": group_name, "node": node, "params": params}


# Single stderr line signaling fallback-parser mode; schema_extractor checks
# it after the returncode gate so stdout stays a pure data channel and the
# signal cannot depend on envelope ordering.
DEGRADED_MARKER = "schema-helper-degraded: docstring_parser unavailable"


def _emit_degraded_marker() -> None:
    """Signal degraded mode on stderr when running on the fallback parser."""
    if _parse_docstring is None:
        print(DEGRADED_MARKER, file=sys.stderr)


def _emit_models() -> None:
    import model  # noqa: F401  triggers @register_model_config discovery
    from utils.config import (
        CompileConfig,
        EarlyStoppingConfig,
        GeneralConfig,
        RunDataConfig,
    )
    from utils.core import MODEL_CONFIGS, get_supported_models

    # Fixed framework groups, in display order: (group_name, RunConfig node, dataclass).
    # Reflected exactly once, BEFORE the per-model loop: a deterministic
    # framework-config failure emits a single error envelope and stops — no
    # per-model re-reflection of the same failing classes. The models
    # envelope is already printed, so the model list still reaches the
    # extractor (an error keyed on None matches no model name).
    models = get_supported_models()
    print(json.dumps({"type": "models", "data": models}))
    try:
        framework_groups = [
            reflect_group(name, node, cls)
            for name, node, cls in (
                ("general", "general", GeneralConfig),
                ("compile", "compile", CompileConfig),
                ("early_stopping", "early_stopping", EarlyStoppingConfig),
                ("data", "data", RunDataConfig),
            )
        ]
    except Exception as e:
        print(json.dumps({"type": "error", "model": None, "error": str(e)}))
        return

    for model_name in models:
        try:
            model_cls = MODEL_CONFIGS.get(model_name)
            groups = [*framework_groups, reflect_group(model_name, "model", model_cls)]
            print(json.dumps({"type": "schema", "model": model_name, "data": groups}))
        except Exception as e:
            print(json.dumps({"type": "error", "model": model_name, "error": str(e)}))


def _emit_preprocess(action: str) -> None:
    from dataclasses import fields as dc_fields

    from utils.config import (
        DownloadConfig,
        GeneralConfig,
        ProcessConfig,
        RunDataConfig,
    )

    # Visible preprocess fields are marked on the dataclass itself via
    # field(metadata={"preprocess_ui": True}), so adding a param only needs
    # that marker — no parallel whitelist here.
    def visible(cls: type) -> set[str]:
        return {f.name for f in dc_fields(cls) if f.metadata.get("preprocess_ui")}

    if action == "download":
        # data_url is resolved from dataset metadata; hide it from the UI.
        groups = [
            reflect_group("download_options", "", DownloadConfig, skip={"data_url"})
        ]
    elif action == "process":
        groups = [
            reflect_group("data", "data", RunDataConfig, only=visible(RunDataConfig)),
            reflect_group(
                "general", "general", GeneralConfig, only=visible(GeneralConfig)
            ),
            reflect_group("extras", "", ProcessConfig),
        ]
    else:
        print(
            json.dumps(
                {
                    "type": "error",
                    "action": action,
                    "error": f"unknown action '{action}'",
                }
            )
        )
        return
    print(json.dumps({"type": "preprocess_schema", "action": action, "data": groups}))


if __name__ == "__main__":
    # Make the repo importable regardless of the caller's cwd (matches
    # PROJECT_ROOT's own derivation in web/backend/config.py). Kept out of
    # module scope so importing the helper (e.g. from pytest) cannot touch
    # the host's sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # Emitted at the single dispatch site so every mode signals degraded
    # runs before any import or envelope output.
    _emit_degraded_marker()
    mode = sys.argv[1] if len(sys.argv) > 1 else "models"
    if mode == "preprocess":
        _emit_preprocess(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        _emit_models()
