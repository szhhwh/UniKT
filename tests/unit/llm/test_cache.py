"""Tests for the SQLite response cache and cache key derivation."""

from pathlib import Path

from utils.llm.cache import SQLiteResponseCache, cache_key


class TestCacheKey:
    def test_same_parts_same_key(self) -> None:
        assert cache_key("generate", "m", None, {"a": 1}) == cache_key(
            "generate", "m", None, {"a": 1}
        )

    def test_different_part_different_key(self) -> None:
        assert cache_key("generate", "m1", None, {}) != cache_key(
            "generate", "m2", None, {}
        )

    def test_dict_key_order_irrelevant(self) -> None:
        assert cache_key({"a": 1, "b": 2}) == cache_key({"b": 2, "a": 1})

    def test_non_json_part_does_not_raise(self) -> None:
        cache_key("generate", "m", None, {"extra": object()})


class TestSQLiteResponseCache:
    def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        cache = SQLiteResponseCache(str(tmp_path / "c.sqlite"))
        cache.put("k1", {"text": "hello", "n": 3})
        assert cache.get_many(["k1", "missing"]) == {"k1": {"text": "hello", "n": 3}}

    def test_missing_keys_absent(self, tmp_path: Path) -> None:
        cache = SQLiteResponseCache(str(tmp_path / "c.sqlite"))
        assert cache.get_many(["nope"]) == {}

    def test_put_overwrites(self, tmp_path: Path) -> None:
        cache = SQLiteResponseCache(str(tmp_path / "c.sqlite"))
        cache.put("k", {"v": 1})
        cache.put("k", {"v": 2})
        assert cache.get_many(["k"]) == {"k": {"v": 2}}

    def test_reopen_persists(self, tmp_path: Path) -> None:
        path = str(tmp_path / "c.sqlite")
        SQLiteResponseCache(path).put("k", {"v": 1})
        assert SQLiteResponseCache(path).get_many(["k"]) == {"k": {"v": 1}}
