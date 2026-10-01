"""Deterministic offline LLM client for tests and demos."""

import hashlib
import math
import random

from utils.core import register_llm_client

from .base import LLMClient, LLMRequest, LLMResponse, TokenLogprob

_VOCAB = ["Yes", "No", "A", "B", "C", "D", "correct", "incorrect"]
_EMBED_DIM = 8


def _digest(*parts: object) -> int:
    """Stable integer seed derived from the request identity."""
    blob = "\x00".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


@register_llm_client("mock")
class MockLLMClient(LLMClient):
    """Hash-based offline double: same request in, same response out.

    Implements the full surface (generate/embed/choice) without network or
    litellm so demos and the CPU test suite run end-to-end.
    """

    def _complete(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Answer each request deterministically from its message hash."""
        out: list[LLMResponse] = []
        for req in requests:
            seed = _digest(self.cfg.model, req.messages)
            rng = random.Random(seed)
            if req.top_logprobs:
                logprobs = self._fake_logprobs(rng)
                text = logprobs[0].token
            else:
                text = f"mock:{seed:012x}"
                logprobs = None
            out.append(
                LLMResponse(
                    text=text,
                    logprobs=logprobs,
                    prompt_tokens=sum(len(str(m)) for m in req.messages) // 4,
                    completion_tokens=max(1, len(text) // 4),
                    cost_usd=0.0,
                )
            )
        return out

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text as a deterministic unit vector from its hash."""
        out: list[list[float]] = []
        for text in texts:
            rng = random.Random(_digest(self.cfg.embedding_model, text))
            vec = [rng.gauss(0.0, 1.0) for _ in range(_EMBED_DIM)]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    @staticmethod
    def _fake_logprobs(rng: random.Random) -> list[TokenLogprob]:
        """Draw a plausible normalized distribution over the fixed vocab."""
        weights = [rng.random() + 0.01 for _ in _VOCAB]
        total = sum(weights)
        pairs = sorted(
            ((t, math.log(w / total)) for t, w in zip(_VOCAB, weights)),
            key=lambda p: -p[1],
        )
        return [TokenLogprob(token=t, logprob=lp) for t, lp in pairs]


__all__ = ["MockLLMClient"]
