"""Tests for the LLM client factory, caching behavior, and usage accounting."""

from pathlib import Path

import pytest

from utils.config import LLMConfig
from utils.core import LLM_CLIENTS
from utils.llm import (
    LLMError,
    LLMRequest,
    MockLLMClient,
    create_llm_client,
)


def _mock_cfg(tmp_path: Path, **overrides: object) -> LLMConfig:
    defaults: dict[str, object] = {
        "provider": "mock",
        "model": "mock-model",
        "embedding_model": "mock-embed",
        "cache_path": str(tmp_path / "cache.sqlite"),
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)  # type: ignore[arg-type]


class TestFactory:
    def test_unconfigured_fails_fast(self) -> None:
        with pytest.raises(ValueError, match=r"llm\.model"):
            create_llm_client(None)
        with pytest.raises(ValueError, match=r"llm\.model"):
            create_llm_client(LLMConfig())

    def test_dispatches_to_provider(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path))
        assert isinstance(client, MockLLMClient)

    def test_unknown_provider_lists_available(
        self, tmp_path: Path, registry_snapshot: None
    ) -> None:
        with pytest.raises(KeyError, match=r"not found.*Available"):
            create_llm_client(_mock_cfg(tmp_path, provider="nope"))

    def test_discovery_indexes_backends(self, registry_snapshot: None) -> None:
        import utils.llm  # noqa: F401

        assert "mock" in LLM_CLIENTS
        assert "litellm" in LLM_CLIENTS


class TestGenerateCache:
    def test_second_call_hits_cache(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path))
        reqs = [LLMRequest(messages=[{"role": "user", "content": "hi"}])]
        first = client.generate(reqs)
        second = client.generate(reqs)
        assert not first[0].cached
        assert second[0].cached
        assert second[0].text == first[0].text
        assert client.usage.requests == 1
        assert client.usage.cache_hits == 1

    def test_cache_shared_across_client_instances(self, tmp_path: Path) -> None:
        cfg = _mock_cfg(tmp_path)
        reqs = [LLMRequest(messages=[{"role": "user", "content": "hi"}])]
        create_llm_client(cfg).generate(reqs)
        second = create_llm_client(cfg).generate(reqs)
        assert second[0].cached

    def test_different_params_miss(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path))
        base = {"role": "user", "content": "hi"}
        client.generate([LLMRequest(messages=[base])])
        other = client.generate([LLMRequest(messages=[base], temperature=0.9)])
        assert not other[0].cached

    def test_cache_disabled_by_null_path(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path, cache_path=None))
        reqs = [LLMRequest(messages=[{"role": "user", "content": "hi"}])]
        client.generate(reqs)
        second = client.generate(reqs)
        assert not second[0].cached
        assert client.usage.requests == 2

    def test_partial_hit_only_sends_misses(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path))
        seen = [LLMRequest(messages=[{"role": "user", "content": msg}]) for msg in "ab"]
        client.generate(seen[:1])
        both = client.generate(seen)
        assert both[0].cached
        assert not both[1].cached
        assert client.usage.requests == 2


class TestEmbed:
    def test_requires_embedding_model(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path, embedding_model=None))
        with pytest.raises(LLMError, match="embedding_model"):
            client.embed(["hi"])

    def test_caches_and_matches_dim(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path))
        first = client.embed(["a", "b"])
        second = client.embed(["a", "b"])
        assert len(first) == 2
        assert len(first[0]) == len(first[1]) == 8
        assert second[0] == first[0]
        assert client.usage.requests == 2
        assert client.usage.cache_hits == 2


class TestUsageAccounting:
    def test_tokens_accumulate_from_transport_only(self, tmp_path: Path) -> None:
        client = create_llm_client(_mock_cfg(tmp_path))
        reqs = [LLMRequest(messages=[{"role": "user", "content": "hello world"}])]
        first = client.generate(reqs)
        client.generate(reqs)
        assert client.usage.prompt_tokens == first[0].prompt_tokens
        assert client.usage.completion_tokens == first[0].completion_tokens
        assert client.usage.cost_usd == 0.0
