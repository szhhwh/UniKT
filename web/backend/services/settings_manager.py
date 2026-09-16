"""Settings manager — JSON-backed persistent settings storage.

Provides load/save operations for application settings (default environment,
concurrency limits, setup status) using an atomic JSON file write pattern.
"""

import contextlib
import json
import os
import tempfile
import threading
from pathlib import Path

from config import DATABASE_PATH

SETTINGS_PATH = DATABASE_PATH.parent / "settings.json"

_DEFAULT_SETTINGS = {
    "default_env_id": None,
    "custom_python_path": None,
    "setup_completed": False,
    "remember_last_env": False,
    "gpu_slots": 1,
}


class SettingsManager:
    """Manages persistent application settings stored as a JSON file.

    Provides atomic writes via a temporary-file-then-rename pattern and
    merges stored values with defaults on every read.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the SettingsManager.

        Args:
            path: Optional path to the settings JSON file. Defaults to
                the standard location next to the database file.
        """
        self._path = path or SETTINGS_PATH
        self._lock = threading.Lock()

    def load(self) -> dict:
        """Load settings from the JSON file, merged with defaults.

        Returns:
            A dict of settings, with missing keys filled from defaults.
        """
        if not self._path.is_file():
            return dict(_DEFAULT_SETTINGS)
        try:
            with open(self._path) as f:
                data = json.load(f)
            return {**_DEFAULT_SETTINGS, **data}
        except (json.JSONDecodeError, OSError):
            return dict(_DEFAULT_SETTINGS)

    def save(self, data: dict) -> None:
        """Save settings to the JSON file using an atomic write.

        Creates the parent directory if needed, merges with existing data,
        writes to a temporary file, and renames atomically.

        Args:
            data: A dict of setting key-value pairs to persist.
        """
        # Hold the lock across load-merge-write so concurrent saves (threadpool
        # endpoints) cannot overwrite each other's field changes.
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            current = self.load()
            merged = {**current, **data}
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent))
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                os.replace(tmp, str(self._path))
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise

    def get_default_env(self) -> str | None:
        """Return the default environment ID.

        Returns:
            The default environment ID, or None.
        """
        return self.load().get("default_env_id")

    def get_custom_python_path(self) -> str | None:
        """Return the custom Python path setting.

        Returns:
            The custom Python path, or None.
        """
        return self.load().get("custom_python_path")

    def set_default_env(
        self, env_id: str, custom_python_path: str | None = None
    ) -> None:
        """Set the default environment and mark setup as completed.

        Args:
            env_id: The environment identifier to set as default.
            custom_python_path: Optional custom Python interpreter path.
        """
        self.save(
            {
                "default_env_id": env_id,
                "custom_python_path": custom_python_path,
                "setup_completed": True,
            }
        )

    def is_setup_completed(self) -> bool:
        """Check whether the initial setup has been completed.

        Returns:
            True if setup has been completed, False otherwise.
        """
        return bool(self.load().get("setup_completed"))

    def get_remember_last_env(self) -> bool:
        """Return whether to remember the last used environment.

        Returns:
            True if the last environment should be remembered.
        """
        return bool(self.load().get("remember_last_env"))

    def set_remember_last_env(self, value: bool) -> None:
        """Set whether to remember the last used environment.

        Args:
            value: The flag value to persist.
        """
        self.save({"remember_last_env": value})

    def get_gpu_slots(self) -> int:
        """Return the persisted per-GPU concurrency slot count.

        Returns:
            The per-GPU slot count, clamped to at least 1. Falls back to 1 if
            the stored value is missing or invalid.
        """
        try:
            return max(1, int(self.load().get("gpu_slots", 1)))
        except (TypeError, ValueError):
            return 1

    def set_gpu_slots(self, value: int) -> int:
        """Persist the per-GPU concurrency slot count.

        Args:
            value: The per-GPU slot count (clamped to >= 1).

        Returns:
            The clamped value that was persisted.
        """
        clamped = max(1, int(value))
        self.save({"gpu_slots": clamped})
        return clamped
