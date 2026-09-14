"""Reflective CLI parser for the RunConfig dataclass tree.

jsonargparse derives CLI flags from the schema (types, ``Literal`` choices, list
nargs, bool), merges ``--config``/``default_config`` yaml under CLI overrides,
and dumps/loads the same shape.

The model node is polymorphic and lazily discovered across multiple environments,
so the model name (``-m`` / ``--experiment.model_name`` / a yaml) is resolved
first and selects the concrete :class:`ModelConfig` subclass before the schema
is built.

``parse_args`` returns a typed :class:`RunConfig` instance. Entry points needing
their own nodes (e.g. ``efficiency.py``) use ``parse_with_extras``. Entry points
that restore an archived run (``evaluate.py``, ``case_analysis.py`` inference)
use :func:`parse_run_archive`, which seeds the schema from the run's
``run_config.yaml`` before reflective CLI overrides apply.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from .run_config import RunConfig, build_run_config_schema


class ConfigParser:
    """Build CLI flags reflectively from the RunConfig schema and parse to a RunConfig."""

    def __init__(
        self,
        prog: str | None = None,
        description: str | None = None,
        extra_nodes: dict[str, type] | None = None,
        default_config: str | Path | None = None,
    ):
        """Store parser identity and optional extension points.

        Args:
            prog: argparse ``prog``.
            description: argparse ``description``.
            extra_nodes: Extra ``{node_name: dataclass}`` nodes (e.g.
                ``{"efficiency": EfficiencyConfig}``) exposed as ``--<node>.<field>``
                flags without polluting the shared RunConfig tree.
            default_config: Path to a yaml loaded as the base config; CLI overrides
                apply on top.
        """
        self.prog = prog
        self.description = description
        self.extra_nodes = dict(extra_nodes) if extra_nodes else {}
        self.default_config = str(default_config) if default_config else None

    def parse_args(self, argv: list[str] | None = None) -> RunConfig:
        """Parse ``argv`` (default ``sys.argv[1:]``) into a :class:`RunConfig` instance."""
        rc, _ = self.parse_with_extras(argv)
        return rc

    def parse_with_extras(
        self, argv: list[str] | None = None
    ) -> tuple[RunConfig, Namespace]:
        """Parse and return ``(RunConfig, full Namespace)`` — Namespace exposes extra nodes."""
        argv = sys.argv[1:] if argv is None else list(argv)
        argv = _expand_short_flags(argv)

        model_name = self._resolve_model_name(argv)
        schema_nodes = {**build_run_config_schema(model_name), **self.extra_nodes}
        parser = self._build_parser(schema_nodes)

        ns = parser.parse_args(argv)
        # An explicit -m/--experiment.model_name wins over any --config/default_config
        # model_name (jsonargparse applies ActionConfigFile by argv position, so a config
        # later in argv could otherwise override the explicit flag).
        explicit = _peek_explicit_model_name(argv)
        if explicit:
            ns["experiment"]["model_name"] = explicit
        rc = _namespace_to_run_config(ns, schema_nodes)
        _require_dataset(rc)
        return rc, ns

    def _resolve_model_name(self, argv: list[str]) -> str:
        """Pass one: find the model name. ``-m`` wins, then ``--config``, then ``default_config``."""
        pre = argparse.ArgumentParser(add_help=False)
        # nargs='?' so a bare ``-m`` (no value) yields None instead of crashing the
        # pre-parser with a truncated usage line.
        pre.add_argument(
            "-m", "--experiment.model_name", dest="model_name", nargs="?", default=None
        )
        pre.add_argument("--config", dest="config", default=None)
        pre_args, _ = pre.parse_known_args(argv)

        if pre_args.model_name:
            return pre_args.model_name
        for source in (pre_args.config, self.default_config):
            if source:
                name = _read_model_name(source)
                if name:
                    return name

        argv_list = argv if argv is not None else sys.argv[1:]
        if "-h" in argv_list or "--help" in argv_list:
            self._print_framework_help()
        from utils.core import get_supported_models

        raise SystemExit(
            "model name is required: pass -m/--experiment.model_name "
            f"(available: {', '.join(get_supported_models())}), or a --config/"
            "default_config yaml carrying experiment.model_name"
        )

    def _build_parser(self, schema_nodes: dict[str, type]) -> ArgumentParser:
        default_files = [self.default_config] if self.default_config else None
        parser = ArgumentParser(
            prog=self.prog,
            description=self.description,
            default_config_files=default_files,
        )
        parser.add_argument("--config", action=ActionConfigFile)
        for node, cls in schema_nodes.items():
            parser.add_class_arguments(cls, node)
        return parser

    def _print_framework_help(self) -> None:
        """Print framework-level help (model-specific flags need ``-m <model> -h``)."""
        from utils.core import get_supported_models

        from .run_config import _FRAMEWORK_NODES

        available = get_supported_models()
        parser = ArgumentParser(
            prog=self.prog,
            description=(
                (self.description or "")
                + f"\nAvailable models: {', '.join(available)}"
                + "\nPass -m <model> -h for model-specific flags."
            ),
        )
        parser.add_argument("--config", action=ActionConfigFile)
        for node, cls in {**_FRAMEWORK_NODES, **self.extra_nodes}.items():
            parser.add_class_arguments(cls, node)
        parser.parse_args(["-h"])  # prints help and exits 0


def _expand_short_flags(argv: list[str]) -> list[str]:
    """Rewrite the two essential short flags to their full jsonargparse forms.

    ``-m X`` -> ``--experiment.model_name X`` and ``-d X`` -> ``--data.dataset X``.
    Kept as the only short flags for ergonomics; everything else uses readable
    dotted flags. Done by argv rewriting so jsonargparse only ever sees full flags.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        long_form, inline = None, None
        if token == "-m":
            long_form = "--experiment.model_name"
        elif token.startswith("-m="):
            inline = "--experiment.model_name=" + token[3:]
        elif token == "-d":
            long_form = "--data.dataset"
        elif token.startswith("-d="):
            inline = "--data.dataset=" + token[3:]
        if inline is not None:
            out.append(inline)
            i += 1
            continue
        if long_form is not None and i + 1 < len(argv):
            out.append(long_form)
            out.append(argv[i + 1])
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _peek_explicit_model_name(argv: list[str]) -> str | None:
    """Read an explicit ``--experiment.model_name`` (post short-flag expansion) from argv."""
    flag = "--experiment.model_name"
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def peek_flag_value(argv: list[str], flag: str) -> str | None:
    """Read the last value of ``flag`` (space or ``=`` form) from argv without parsing.

    Entry points need ``--<node>.run_dir`` before :class:`ConfigParser` runs
    (the archive path feeds ``default_config``). Last occurrence wins, matching
    argparse's semantics for repeated flags.
    """
    value = None
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif token.startswith(flag + "="):
            value = token.split("=", 1)[1]
    return value


def parse_run_archive(
    argv: list[str] | None,
    *,
    prog: str,
    description: str,
    entry_node: str,
    entry_cls: type,
) -> tuple[RunConfig, Any, Path]:
    """Parse an entry point that restores a RunConfig from a run archive.

    Shared skeleton for ``evaluate.py`` and ``case_analysis.py`` inference: the
    archived ``run_config.yaml`` seeds the RunConfig, any reflective CLI flag
    overrides it, and entry-point knobs come from the typed ``entry_cls`` node.

    Args:
        argv: CLI tokens (default ``sys.argv[1:]``).
        prog: argparse ``prog`` used in error messages.
        description: argparse ``description`` shown by ``--help``.
        entry_node: Extra-node name; the run directory is read from
            ``--<entry_node>.run_dir``.
        entry_cls: Dataclass exposed as ``--<entry_node>.*`` flags; must
            declare a ``run_dir`` field.

    Returns:
        ``(rc, entry_cfg, resolved_run_dir)``.

    Raises:
        SystemExit: ``run_dir`` missing (``-h``/``--help`` still prints the
            framework reference and exits 0), an explicit model flag present,
            or the archive file absent.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    run_dir_flag = f"--{entry_node}.run_dir"
    run_dir = peek_flag_value(argv, run_dir_flag)
    if run_dir is None:
        if "-h" in argv or "--help" in argv:
            # Framework reference without a bound model; exits 0 from inside.
            ConfigParser(
                prog=prog,
                description=description,
                extra_nodes={entry_node: entry_cls},
            ).parse_with_extras(argv)
        raise SystemExit(
            f"{prog}: {run_dir_flag} is required (point it at a trained run directory)"
        )
    _reject_model_flags(argv, prog=prog, run_dir_flag=run_dir_flag)
    archive = Path(run_dir) / "run_config.yaml"
    if not archive.exists():
        raise SystemExit(f"{prog}: run_config.yaml not found in {run_dir}")
    rc, ns = ConfigParser(
        prog=prog,
        description=description,
        extra_nodes={entry_node: entry_cls},
        default_config=archive,
    ).parse_with_extras(argv)
    entry_cfg = build_node(entry_cls, ns[entry_node])
    return rc, entry_cfg, Path(run_dir).resolve()


def _reject_model_flags(argv: list[str], prog: str, run_dir_flag: str) -> None:
    """Archive mode rebuilds the model from the run_config.yaml it locates."""
    for token in argv:
        if token.startswith("-m") or token.startswith("--experiment.model"):
            raise SystemExit(
                f"{prog}: {run_dir_flag} cannot be combined with "
                "-m/--experiment.model_name (the model is restored from the archive)"
            )


def _require_dataset(rc: RunConfig) -> None:
    """Fail fast with a clear message when no dataset is selected.

    Without this, an empty ``rc.data.dataset`` reaches ``get_data_source`` late
    (after the run directory is created) as an opaque "Unsupported dataset:" error.
    """
    if rc.data.dataset:
        return
    from utils.core import get_supported_datasets

    raise SystemExit(
        "dataset is required: pass -d/--data.dataset "
        f"(available: {', '.join(get_supported_datasets())})"
    )


def _read_model_name(config_path: str) -> str | None:
    """Read ``experiment.model_name`` from a yaml config file (no schema needed)."""
    try:
        with Path(config_path).open() as f:
            data = yaml.safe_load(f) or {}
        return data.get("experiment", {}).get("model_name")
    except (OSError, yaml.YAMLError):
        return None


def _namespace_to_run_config(ns: Namespace, schema_nodes: dict[str, type]) -> RunConfig:
    """Construct a :class:`RunConfig` from a jsonargparse Namespace.

    Only the standard RunConfig nodes are built; entry-point ``extra_nodes`` stay
    in the Namespace (returned alongside by ``parse_with_extras``).
    """
    rc_fields = {f.name for f in fields(RunConfig)}
    node_instances: dict[str, Any] = {}
    for node, cls in schema_nodes.items():
        if node in rc_fields:
            node_instances[node] = build_node(cls, ns[node])
    return RunConfig(**node_instances)


def build_node(cls: type, node_ns: Namespace) -> Any:
    """Instantiate a (possibly nested) config dataclass from its Namespace slice.

    Scalar fields map straight through; a dataclass-valued field whose slice is
    still a Namespace recurses. The efficiency node is the repo's first nested
    config (general + per-stage sub-nodes); RunConfig nodes have only scalar
    fields, so this is a no-op for them.
    """
    from dataclasses import is_dataclass
    from typing import get_type_hints

    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        val = node_ns[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(val, Namespace):
            kwargs[f.name] = build_node(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


__all__ = [
    "ConfigParser",
    "build_node",
    "parse_run_archive",
    "peek_flag_value",
]
