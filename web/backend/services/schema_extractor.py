"""Schema extractor — model discovery and parameter schema retrieval.

Runs a helper subprocess that introspects available training models and their
CLI parameter schemas, returning structured data for the schemas API.
"""

import json
import logging
import subprocess
import sys
import threading
from collections.abc import Iterator

from config import PROJECT_ROOT
from schemas import ModelSchemaResponse, ParamField, ParamGroup

from services._schema_helper import DEGRADED_MARKER
from services.python_env import EnvironmentNotConfigured, PythonEnvManager

logger = logging.getLogger(__name__)

HELPER_SCRIPT = str(PROJECT_ROOT / "web" / "backend" / "services" / "_schema_helper.py")


def _warn_if_degraded(result: subprocess.CompletedProcess) -> None:
    """Log when the helper ran on its fallback docstring parser.

    The helper signals this on stderr (DEGRADED_MARKER); checking it once
    here, after the returncode gate, keeps stdout a pure data channel and
    frees envelope consumers from ordering concerns.
    """
    if DEGRADED_MARKER in result.stderr:
        logger.warning(
            "Schema helper environment lacks docstring_parser (conda/custom "
            "interpreter?); schema helps fall back to the built-in parser"
        )


def _iter_envelopes(result: subprocess.CompletedProcess) -> Iterator[dict]:
    """Yield each valid dict JSON envelope line from helper stdout.

    Skips blank, non-JSON, and non-dict lines — dependencies may print bare
    JSON to stdout during the helper's imports.
    """
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            yield entry


def _parse_group(data: dict) -> ParamGroup:
    """Parse a raw parameter group dict into a ParamGroup model.

    Args:
        data: Raw dict with ``group_name`` and ``params`` keys.

    Returns:
        A ParamGroup instance with parsed ParamField entries.
    """
    fields = {}
    for name, cfg in data["params"].items():
        fields[name] = ParamField(
            type=cfg.get("type", "str"),
            default=cfg.get("default"),
            help=cfg.get("help", ""),
            required=cfg.get("required", False),
            choices=cfg.get("choices"),
            short=cfg.get("short"),
            nargs=cfg.get("nargs"),
            optuna=cfg.get("optuna"),
        )
    return ParamGroup(
        group_name=data["group_name"], node=data.get("node"), params=fields
    )


class SchemaExtractor:
    """Extracts model names and parameter schemas by running a helper script.

    Caches results after the first successful run to avoid repeated subprocess
    invocations.

    Args:
        env_manager: A PythonEnvManager instance (or None) used to resolve
            the Python command for the helper script.
    """

    def __init__(self, env_manager: PythonEnvManager | None) -> None:
        """Initialize the SchemaExtractor.

        Args:
            env_manager: PythonEnvManager (or compatible) for command resolution.
        """
        self._env_manager = env_manager
        self._models: list[str] = []
        self._schemas: dict[str, list[dict]] = {}
        self._preprocess_schemas: dict[str, list[dict] | None] = {}
        # Single-flight + staleness guard for the lazy per-action preprocess
        # extractions (all under _cond): loading tracks in-flight actions so
        # concurrent misses share one subprocess; gen invalidates results a
        # reset_cache() raced past.
        self._preprocess_loading: set[str] = set()
        self._preprocess_gen = 0
        self._loaded: bool = False
        self._loading: bool = False
        self._cond = threading.Condition()

    def _resolve_base_cmd(self) -> list[str]:
        """Resolve the base Python command for the helper script.

        Uses the wizard-configured default environment.

        Raises:
            EnvironmentNotConfigured: If no default environment is configured.
        """
        if self._env_manager:
            return self._env_manager.resolve_command()
        return [sys.executable]

    def _extract(self) -> tuple[list[str], dict[str, list[dict]]] | None:
        """Run the helper subprocess once, returning (models, schemas) or None.

        Pure of instance state so the caller can commit the result under the
        lock. Errored models are excluded from the returned list.
        """
        result = self._run_helper_subprocess([], "Model")
        if result is None:
            return None
        all_models: list[str] | None = None
        errored: set[str] = set()
        schemas: dict[str, list[dict]] = {}
        # .get payload guards: dependencies may print bare JSON to stdout
        # during the helper's imports, and subscripts on a noise dict whose
        # "type" collides would raise out of _extract. Last-wins assignment
        # (matching main): the helper's imports run before it prints, so a
        # colliding noise line realistically lands BEFORE the real envelope
        # and last-wins keeps the real one.
        for entry in _iter_envelopes(result):
            etype = entry.get("type")
            if etype == "models":
                data = entry.get("data")
                if isinstance(data, list):
                    all_models = data
            elif etype == "schema":
                model, data = entry.get("model"), entry.get("data")
                if model and isinstance(data, list):
                    schemas[model] = data
            elif etype == "error":
                model = entry.get("model")
                if model:
                    errored.add(model)
        if all_models is None:
            # Exit 0 but no models envelope: truncated or malformed output.
            logger.error(
                "Schema helper emitted no model list — %s",
                result.stderr.strip() or "(no stderr)",
            )
            return None
        if errored and set(all_models) <= errored:
            # Every model errored (e.g. a framework-config default turned
            # non-JSON-serializable). Commit the empty result with a loud
            # log line rather than failing uncached — an uncached failure
            # would re-run the full torch-importing subprocess per request.
            logger.error(
                "Schema helper errored on every model — %s",
                result.stderr.strip() or "(no stderr)",
            )
        return [m for m in all_models if m not in errored], schemas

    def _run_helper_subprocess(
        self, args: list[str], label: str
    ) -> subprocess.CompletedProcess | None:
        """Run the schema helper subprocess; None on any failure.

        Shared by the models and preprocess paths so env resolution, timeouts,
        and non-zero-exit logging live in exactly one place.
        """
        try:
            cmd = [*self._resolve_base_cmd(), HELPER_SCRIPT, *args]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=str(PROJECT_ROOT)
            )
        except EnvironmentNotConfigured as e:
            # No env configured (user skipped the setup wizard); callers
            # return None so the next call retries once the wizard sets one.
            logger.error("%s schema extraction skipped — %s", label, e)
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Transient failure; callers return None so the next call retries.
            return None
        except (ValueError, UnicodeDecodeError) as e:
            # Configuration-class failures (unknown env-type prefix from
            # resolve_command, non-UTF-8 helper stdout): degrade with a log
            # line on BOTH paths — without this the preprocess route escaped
            # as an unhandled 500 while the models path only logged.
            logger.error("%s schema extraction failed — %s", label, e)
            return None
        if result.returncode != 0:
            # Without this an empty stdout would parse to ([], {}) and be
            # cached as a successful load, short-circuiting every later call.
            logger.error(
                "%s schema helper exited %d — %s",
                label,
                result.returncode,
                result.stderr.strip() or "(no stderr)",
            )
            return None
        _warn_if_degraded(result)
        return result

    def _run_helper(self) -> None:
        """Load model list/schemas via the helper subprocess (single-flight).

        The import-heavy subprocess runs OUTSIDE the lock so concurrent readers
        (task dispatch, schemas API) are not serialized behind it; later
        callers wait for the in-flight load instead of spawning their own.
        """
        with self._cond:
            if self._loaded:
                return
            if self._loading:
                while self._loading:
                    self._cond.wait()
                return
            self._loading = True
        result = None
        try:
            result = self._extract()
        except Exception:
            # Never leave _loading stuck True: every later call would block
            # on cond.wait() forever (schemas API, registry refresh, dispatch).
            logger.exception("Schema extraction crashed")
        finally:
            # finally (not except-then-continue): even a BaseException must
            # release the single-flight slot on its way out.
            with self._cond:
                if result is not None:
                    # Atomic replacement: readers always see a complete container.
                    self._models, self._schemas = result
                    self._loaded = True
                self._loading = False
                self._cond.notify_all()

    def reset_cache(self) -> None:
        """Mark caches stale so the next access re-extracts them.

        Containers are left in place: until the next _run_helper atomically
        replaces them, readers still get the last valid snapshot instead of an
        empty one. Waits for any in-flight load first so its result can't
        un-clear the cache after we reset.
        """
        with self._cond:
            while self._loading:
                self._cond.wait()
            self._loaded = False
            self._preprocess_schemas.clear()
            # Invalidate in-flight preprocess extractions too: they may be
            # mid-subprocess right now and would otherwise commit their
            # pre-refresh result after this clear.
            self._preprocess_gen += 1

    def list_models(self) -> list[str]:
        """Return the list of available model names.

        Returns:
            A list of model name strings.
        """
        self._run_helper()
        return self._models

    def get_model_schema(self, model_name: str) -> ModelSchemaResponse:
        """Return the parameter schema for a specific model.

        Args:
            model_name: The model name to look up.

        Returns:
            A ModelSchemaResponse with parameter groups and fields.

        Raises:
            KeyError: If the model name is not found.
        """
        self._run_helper()
        raw_groups = self._schemas.get(model_name)
        if raw_groups is None:
            raise KeyError(f"Model '{model_name}' not found")
        groups = [_parse_group(g) for g in raw_groups]
        return ModelSchemaResponse(model_name=model_name, param_groups=groups)

    def get_field_routes(self, model_name: str) -> dict[str, str]:
        """Return a {field_name: RunConfig node} map for routing flat params.

        Derived from the cached schema so the backend can route frontend params
        into RunConfig nodes without importing the model config under torch.

        Args:
            model_name: The model name to look up.

        Returns:
            A dict mapping each parameter field name to its RunConfig node.

        Raises:
            KeyError: If the model name is not found.
        """
        self._run_helper()
        raw_groups = self._schemas.get(model_name)
        if raw_groups is None:
            raise KeyError(f"Model '{model_name}' not found")
        routes: dict[str, str] = {}
        for group in raw_groups:
            node = group.get("node")
            if not node:
                continue
            for field_name in group.get("params", {}):
                routes[field_name] = node
        return routes

    def get_field_defaults(self, model_name: str) -> dict[str, object]:
        """Return a {field_name: default_value} map for filtering unchanged params.

        Args:
            model_name: The model name to look up.

        Returns:
            A dict mapping each parameter field name to its default value.

        Raises:
            KeyError: If the model name is not found.
        """
        self._run_helper()
        raw_groups = self._schemas.get(model_name)
        if raw_groups is None:
            raise KeyError(f"Model '{model_name}' not found")
        defaults: dict[str, object] = {}
        for group in raw_groups:
            for field_name, cfg in group.get("params", {}).items():
                defaults[field_name] = cfg.get("default")
        return defaults

    def _run_preprocess_helper(self, action: str) -> list[dict] | None:
        """Run ``_schema_helper preprocess <action>`` once and cache raw groups.

        Per-action and lazy (runs on first request for that action), unlike the
        eager model helper. Returns None on missing env / subprocess failure so
        callers can raise KeyError. Single-flight per action: concurrent misses
        share one subprocess, and a result whose generation was invalidated by
        a reset_cache() mid-run is not committed.
        """
        with self._cond:
            waited = False
            while True:
                cached = self._preprocess_schemas.get(action)
                if cached is not None:
                    return cached
                if action not in self._preprocess_loading:
                    if waited:
                        # The extraction we waited for failed; fail this
                        # request too instead of serially re-running the
                        # subprocess behind it (N waiters would mean N x 60s).
                        return None
                    self._preprocess_loading.add(action)
                    gen = self._preprocess_gen
                    break
                # Another thread is extracting this action; wait for it.
                waited = True
                self._cond.wait()
        raw: list[dict] | None = None
        try:
            result = self._run_helper_subprocess(["preprocess", action], "Preprocess")
            if result is not None:
                for entry in _iter_envelopes(result):
                    if (
                        entry.get("type") == "preprocess_schema"
                        and entry.get("action") == action
                    ):
                        data = entry.get("data")
                        if isinstance(data, list):
                            raw = data
                            break
        finally:
            # Only a successful, still-current extraction is cached; a None
            # result is not, so the next call retries instead of raising
            # KeyError forever until restart. A result invalidated by a
            # reset_cache() mid-run is dropped AND served as a miss, so the
            # leader and its waiters see the same outcome instead of the
            # leader getting stale data while waiters 404.
            with self._cond:
                self._preprocess_loading.discard(action)
                if raw is not None and gen != self._preprocess_gen:
                    raw = None
                if raw is not None:
                    self._preprocess_schemas[action] = raw
                self._cond.notify_all()
        return raw

    def get_preprocess_schema(self, action: str) -> list[ParamGroup]:
        """Return the parameter schema for a preprocess action (download/process).

        Raises:
            KeyError: If the action is unknown or extraction failed.
        """
        raw_groups = self._run_preprocess_helper(action)
        if raw_groups is None:
            raise KeyError(f"Preprocess action '{action}' not found")
        return [_parse_group(g) for g in raw_groups]

    def get_preprocess_field_routes(self, action: str) -> dict[str, str]:
        """Return ``{field_name: node}`` for routing flat preprocess params.

        ``node`` is ``""`` for flat flags (--force/--extra) and ``data``/``general``
        for nested ones; the command builder treats an empty node as flat and
        skips keys missing from the map entirely.
        """
        raw_groups = self._run_preprocess_helper(action)
        if raw_groups is None:
            raise KeyError(f"Preprocess action '{action}' not found")
        routes: dict[str, str] = {}
        for group in raw_groups:
            node = group.get("node") or ""
            for field_name in group.get("params", {}):
                routes[field_name] = node
        return routes

    def get_preprocess_field_defaults(self, action: str) -> dict[str, object]:
        """Return ``{field_name: default}`` for filtering unchanged preprocess params."""
        raw_groups = self._run_preprocess_helper(action)
        if raw_groups is None:
            raise KeyError(f"Preprocess action '{action}' not found")
        defaults: dict[str, object] = {}
        for group in raw_groups:
            for field_name, cfg in group.get("params", {}).items():
                defaults[field_name] = cfg.get("default")
        return defaults
