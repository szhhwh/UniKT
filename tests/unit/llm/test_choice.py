"""Tests for choice() first-token probability normalization."""

import math
from pathlib import Path

import pytest

from utils.config import LLMConfig
from utils.llm import (
    ChoiceRequest,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    TokenLogprob,
    create_llm_client,
)


def _resp(pairs: list[tuple[str, float]]) -> LLMResponse:
    return LLMResponse(
        text="",
        logprobs=[TokenLogprob(token=t, logprob=math.log(p)) for t, p in pairs],
    )


class TestNormalizeChoice:
    def test_simple_normalization(self) -> None:
        probs = LLMClient._normalize_choice(
            ["Yes", "No"], _resp([("Yes", 0.75), ("No", 0.25)])
        )
        assert probs["Yes"] == pytest.approx(0.75)
        assert probs["No"] == pytest.approx(0.25)

    def test_leading_space_variant_merges(self) -> None:
        probs = LLMClient._normalize_choice(
            ["Yes", "No"], _resp([(" Yes", 0.5), ("Yes", 0.25), ("No", 0.25)])
        )
        assert probs["Yes"] == pytest.approx(0.75)
        assert probs["No"] == pytest.approx(0.25)

    def test_unrelated_mass_is_renormalized_away(self) -> None:
        probs = LLMClient._normalize_choice(
            ["Yes", "No"], _resp([("Maybe", 0.6), ("Yes", 0.3), ("No", 0.1)])
        )
        assert probs["Yes"] == pytest.approx(0.75)
        assert probs["No"] == pytest.approx(0.25)

    def test_zero_mass_raises_with_candidates(self) -> None:
        with pytest.raises(LLMError, match="Maybe"):
            LLMClient._normalize_choice(["Yes", "No"], _resp([("Maybe", 1.0)]))

    def test_missing_logprobs_raises(self) -> None:
        with pytest.raises(LLMError, match="logprobs"):
            LLMClient._normalize_choice(["Yes", "No"], LLMResponse(text="Yes"))

    def test_strip_colliding_options_raise(self) -> None:
        # "Yes" and " Yes" would each match the same token mass and split it.
        with pytest.raises(LLMError, match="collide"):
            LLMClient._normalize_choice(
                ["Yes", " Yes"], _resp([("Yes", 0.75), ("No", 0.25)])
            )


class TestChoiceEndToEnd:
    def test_mock_choice_is_deterministic_and_normalized(self, tmp_path: Path) -> None:
        cfg = LLMConfig(
            provider="mock",
            model="mock-model",
            cache_path=str(tmp_path / "c.sqlite"),
        )
        client = create_llm_client(cfg)
        reqs = [
            ChoiceRequest(
                messages=[{"role": "user", "content": "will he solve it?"}],
                options=["Yes", "No"],
            )
            for _ in range(3)
        ]
        results = client.choice(reqs)
        for probs in results:
            assert set(probs) == {"Yes", "No"}
            assert sum(probs.values()) == pytest.approx(1.0)
        assert results[0] == results[1] == results[2]

    def test_choice_shares_generate_cache(self, tmp_path: Path) -> None:
        cfg = LLMConfig(
            provider="mock",
            model="mock-model",
            cache_path=str(tmp_path / "c.sqlite"),
        )
        client = create_llm_client(cfg)
        messages = [{"role": "user", "content": "q"}]
        client.choice([ChoiceRequest(messages=messages, options=["Yes", "No"])])
        # Same messages, same resolved params (max_tokens=1, top_logprobs=20).
        again = client.generate(
            [LLMRequest(messages=messages, max_tokens=1, top_logprobs=20)]
        )
        assert again[0].cached
