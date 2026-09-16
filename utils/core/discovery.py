"""Static registration discovery: scan source code (no imports) to build a name→module lazy index.

Scans for ``@register_<role>("name")`` decorators and populates the corresponding
registry's ``index``. ``keys()`` then lists all components at startup while
component code is only imported on ``get()`` (lazy loading, multi-environment safe).

The decorator→registry mapping is self-described: each :class:`UniversalRegistry`
declares its own ``decorator_name``, which discovery reads.

Only the uniform ``@register_<role>("literal")`` form is recognized; non-literal
arguments (variables, expressions) are ignored.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .logger import get_logger
from .registry import UniversalRegistry

_logger = get_logger(__name__)


def _decorator_map() -> dict[str, UniversalRegistry]:
    """Build ``{decorator_name: registry}`` from the registries themselves."""
    return {
        r.decorator_name: r
        for r in UniversalRegistry._all_registries
        if r.decorator_name
    }


def discover_registrations(root: str | Path, package: str) -> None:
    """Scan all ``.py`` files under ``root`` (excluding ``__init__``) and write decorator registrations into the lazy index.

    Args:
        root: Directory to scan (pass ``__file__``'s parent directory, independent of the current working directory).
        package: Dotted module name for the directory (e.g. ``"model"``, ``"utils.data_process"``).
    """
    root_path = Path(root)
    decorators = _decorator_map()
    found = 0
    for py in root_path.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        module_path = _to_module_path(py, root_path, package)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            _logger.warning("Skip %s: %s", py, exc)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    matched = _target_registry(dec, decorators)
                    if matched is not None:
                        registry, registered_name = matched
                        registry.index(registered_name, module_path)
                        found += 1
    _logger.debug("discovery: in %s, found %d registrations", root, found)


def _to_module_path(py: Path, root: Path, package: str) -> str:
    rel = py.relative_to(root).with_suffix("")
    return ".".join((package, *rel.parts))


def _target_registry(
    dec: ast.expr, decorators: dict[str, UniversalRegistry]
) -> tuple[UniversalRegistry, str] | None:
    """Identify ``@register_<role>("literal")``; return registry + registered name.

    Returns ``None`` when the decorator does not match the expected shape.
    """
    if not isinstance(dec, ast.Call) or not dec.args:
        return None
    if not (
        isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str)
    ):
        return None
    registry = decorators.get(dec.func.id) if isinstance(dec.func, ast.Name) else None
    if registry is None:
        return None
    return registry, dec.args[0].value
