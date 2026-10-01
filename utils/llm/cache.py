"""SQLite content-addressed cache for LLM responses.

Keys hash the full request identity (method, model, endpoint, payload,
sampling parameters), so a single cache file is safely shared across runs,
folds, and concurrent processes (WAL journal + busy timeout).
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def cache_key(*parts: Any) -> str:
    """Return a stable sha256 key over the canonical JSON of ``parts``.

    ``sort_keys`` canonicalizes nested dicts; ``default=str`` keeps
    non-JSON-native values (e.g. provider-specific extras) hashable instead
    of raising.

    Args:
        parts: Components identifying the cached entry.

    Returns:
        Hex digest of the canonical JSON serialization.
    """
    blob = json.dumps(list(parts), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SQLiteResponseCache:
    """Thread-safe key/payload store for cached LLM responses."""

    def __init__(self, path: str) -> None:
        """Open (or create) the cache database.

        Args:
            path: SQLite database file (parent directories are created).
        """
        db = Path(path)
        db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db), check_same_thread=False)
        # WAL + busy timeout: parallel runs (e.g. optuna sweeps) write
        # concurrently without failing fast on "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """Look up cached payloads.

        Args:
            keys: Cache keys produced by :func:`cache_key`.

        Returns:
            Mapping of key to payload for the keys that are present.
        """
        found: dict[str, dict[str, Any]] = {}
        if not keys:
            return found
        with self._lock:
            for key in keys:
                row = self._conn.execute(
                    "SELECT payload FROM responses WHERE key = ?", (key,)
                ).fetchone()
                if row is not None:
                    found[key] = json.loads(row[0])
        return found

    def put(self, key: str, payload: dict[str, Any]) -> None:
        """Store one payload, committed immediately so crashes lose nothing.

        Args:
            key: Cache key produced by :func:`cache_key`.
            payload: JSON-serializable payload.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, payload) VALUES (?, ?)",
                (key, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()


__all__ = ["SQLiteResponseCache", "cache_key"]
