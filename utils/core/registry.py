"""Unified registry system: decorator-based registration with lazy static discovery.

Each registry maintains two tables:

- ``_registry``: name -> class, populated at module import time by the
  ``@register_<role>(...)`` decorator.
- ``_index``: name -> module path, populated by :mod:`utils.core.discovery`
  via static source scanning **without importing** the module.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

# Unbound: solved per-decoration from the decorated class (bound=type breaks
# the Callable[[type[T]], type[T]] decorator-factory idiom under mypy).
T = TypeVar("T")


class UniversalRegistry:
    """A generic registry supporting decorator registration and lazy static discovery.

    Example:
        >>> TRAINERS = UniversalRegistry("trainers")
        >>> @register_trainer("GIKT")
        ... class GIKTTrainer: ...
        >>> TRAINERS.get("GIKT")

    Attributes:
        _name: Human-readable name of this registry (e.g. ``"trainers"``).
        _registry: Mapping from registered name to class.
        _index: Mapping from registered name to dotted module path for lazy
            loading.
    """

    # Roll-call of every registry instance, iterated by discovery. Mutable by design.
    _all_registries: list["UniversalRegistry"] = []  # noqa: RUF012

    def __init__(self, name: str, decorator_name: str | None = None):
        """Initialize the registry.

        Args:
            name: Human-readable registry name (e.g. ``"trainers"``).
            decorator_name: The ``@register_<role>`` function name that populates
                this registry via static discovery. ``None`` marks an import-time-only
                registry (not discovered), e.g. ``METRIC_LOGGERS``.
        """
        self._name = name
        self.decorator_name = decorator_name
        self._registry: dict[str, type] = {}
        self._index: dict[str, str] = {}
        UniversalRegistry._all_registries.append(self)

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        """Return a decorator that binds a class to ``name`` at import time.

        Args:
            name: Registration name. If ``None``, the class name is used.

        Raises:
            KeyError: The name is already bound to a **different** class.

        Returns:
            A decorator that registers the class.
        """

        def decorator(cls: type[T]) -> type[T]:
            n = name if name is not None else cls.__name__
            prev = self._registry.get(n)
            if prev is not None and prev is not cls:
                raise KeyError(f"'{n}' already registered in '{self._name}'")
            self._registry[n] = cls
            self._index.pop(n, None)  # Instantiated, lazy index no longer needed
            return cls

        return decorator

    def index(self, name: str, module_path: str) -> None:
        """Record the module path of an entry **without importing** the module.

        This is the entry point for static discovery.

        Args:
            name: Registration name.
            module_path: Dotted module path where the name resides
                (e.g. ``model.GIKT.GIKT_trainer``).

        Raises:
            KeyError: The same name was indexed from two different modules.
        """
        prev = self._index.get(name)
        if prev is not None and prev != module_path:
            raise KeyError(
                f"'{name}' indexed twice in '{self._name}': {prev} vs {module_path}"
            )
        self._index.setdefault(name, module_path)

    def clear(self) -> None:
        """Drop all loaded entries and the lazy index.

        Returns the registry to its empty state so the next
        :func:`discover_registrations` rebuilds the index from the current
        source tree — used by the web backend to pick up components added or
        removed on disk after the process started.
        """
        self._registry.clear()
        self._index.clear()

    def get(self, name: str) -> type:
        """Look up a class by name; lazy-import the module if necessary.

        Args:
            name: Registration name.

        Raises:
            KeyError: The name is not registered.

        Returns:
            The registered class.
        """
        cls = self._registry.get(name)
        if cls is not None:
            return cls
        module_path = self._index.get(name)
        if module_path is not None:
            import importlib

            importlib.import_module(
                module_path
            )  # Triggers decorator -> populates _registry
            cls = self._registry.get(name)
            if cls is not None:
                return cls
        raise KeyError(
            f"'{name}' not found in '{self._name}'. Available: {', '.join(self.keys())}"
        )

    def keys(self) -> list[str]:
        """Return all registered names (loaded and lazy-indexed, deduplicated).

        Returns:
            A list of all known registered names.
        """
        seen = list(self._registry.keys())
        seen += [k for k in self._index if k not in self._registry]
        return seen

    def __bool__(self) -> bool:
        """Return ``True`` if the registry has any entries (loaded or indexed).

        This makes ``if not REGISTRY`` behave as expected for empty registries.
        """
        return bool(self._registry) or bool(self._index)

    def __len__(self) -> int:
        """Return the number of unique registered entries (loaded + indexed)."""
        return len(self.keys())

    def __contains__(self, name: object) -> bool:
        """Check whether a name is registered.

        Args:
            name: Registration name to look up.

        Returns:
            ``True`` if the name is in the registry or the index.
        """
        return name in self._registry or name in self._index

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names (loaded and lazy-indexed, deduplicated).

        Returns:
            An iterator over all known registered names.
        """
        return iter(self.keys())

    def __repr__(self) -> str:
        """Return a string representation of the registry.

        Returns:
            A string showing the registry name and its registered items.
        """
        return f"UniversalRegistry('{self._name}', items={self.keys()})"


# ============================================================================
# Global registries
# ============================================================================

TRAINERS = UniversalRegistry("trainers", decorator_name="register_trainer")
MODEL_CONFIGS = UniversalRegistry(
    "model_configs", decorator_name="register_model_config"
)
DATA_SOURCES = UniversalRegistry("data_sources", decorator_name="register_data_source")
ANALYZERS = UniversalRegistry("analyzers", decorator_name="register_analyzer")
METRIC_LOGGERS = UniversalRegistry("metric_loggers")  # import-time only, not discovered
EFFICIENCY_STAGES = UniversalRegistry(
    "efficiency_stages", decorator_name="register_efficiency_stage"
)
METRICS = UniversalRegistry("metrics", decorator_name="register_metric")
CASE_SINKS = UniversalRegistry("case_sinks", decorator_name="register_case_sink")
CASE_SELECTORS = UniversalRegistry(
    "case_selectors", decorator_name="register_case_selector"
)
CASE_VISUALIZERS = UniversalRegistry(
    "case_visualizers", decorator_name="register_case_visualizer"
)


# ============================================================================
# Convenience decorators: unified @register_<role>("name") vocabulary
# ============================================================================


def register_trainer(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a trainer into ``TRAINERS``.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``TRAINERS``.
    """
    return TRAINERS.register(name)


def register_model_config(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a per-model ``ModelConfig`` dataclass into ``MODEL_CONFIGS``.

    Combines ``@dataclass`` transformation and registry binding into one
    decorator: apply ``@register_model_config("Name")`` directly on a
    :class:`~utils.config.run_config.ModelConfig` subclass.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that applies ``@dataclass`` then registers the result.
    """
    register = MODEL_CONFIGS.register(name)

    def decorator(cls: type[T]) -> type[T]:
        return register(dataclass(cls))

    return decorator


def register_data_source(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a data source into ``DATA_SOURCES``.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``DATA_SOURCES``.
    """
    return DATA_SOURCES.register(name)


def register_analyzer(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register an analysis case into ``ANALYZERS``.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``ANALYZERS``.
    """
    return ANALYZERS.register(name)


def register_metric_logger(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a metric logging backend into ``METRIC_LOGGERS``.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``METRIC_LOGGERS``.
    """
    return METRIC_LOGGERS.register(name)


def register_efficiency_stage(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register an efficiency benchmark stage into ``EFFICIENCY_STAGES``.

    Stages are auto-discovered from ``utils/efficiency/stages/``; drop a file
    there and decorate the class — no manual registration elsewhere.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``EFFICIENCY_STAGES``.
    """
    return EFFICIENCY_STAGES.register(name)


def register_metric(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a metric into ``METRICS``.

    Drop a ``@register_metric("name")`` class under
    :mod:`utils.training.metrics` and it is auto-discovered — no manual
    registration elsewhere.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``METRICS``.
    """
    return METRICS.register(name)


def register_case_sink(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a case analysis result sink into ``CASE_SINKS``.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``CASE_SINKS``.
    """
    return CASE_SINKS.register(name)


def register_case_selector(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a case analysis user selector into ``CASE_SELECTORS``.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``CASE_SELECTORS``.
    """
    return CASE_SELECTORS.register(name)


def register_case_visualizer(name: str | None = None) -> Callable[[type[T]], type[T]]:
    """Register a case analysis visualizer into ``CASE_VISUALIZERS``.

    Args:
        name: Optional registration name. Defaults to the class name.

    Returns:
        A decorator that registers the class with ``CASE_VISUALIZERS``.
    """
    return CASE_VISUALIZERS.register(name)
