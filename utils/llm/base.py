"""LLM client abstraction: capability-oriented, batch-in/batch-out.

Transport (provider routing, retries, endpoints) is delegated to concrete
subclasses — the production one wraps litellm. This module owns the
KT-specific concerns: persistent content-addressed caching, usage
accounting, and the choice-scoring primitive used by prompting-based KT.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, cast

from utils.config.run_config import LLMConfig
from utils.core import get_logger
from utils.core.registry import LLM_CLIENTS

from .cache import SQLiteResponseCache, cache_key

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM request cannot be served or its response parsed."""


@dataclass
class LLMRequest:
    """One generation request; per-request overrides of LLMConfig defaults.

    Args:
        messages: OpenAI-style ``[{"role": ..., "content": ...}]`` list.
            ``content`` may be a multimodal block list (e.g. question images)
            and is passed through to the transport unchanged.
        temperature: Override of ``llm.temperature``.
        max_tokens: Override of ``llm.max_tokens``.
        top_logprobs: Request the first-token distribution with this many
            candidates; populates :attr:`LLMResponse.logprobs`.
        seed: Sampling seed; part of the cache key when set.
        response_format: Provider passthrough, e.g. ``{"type": "json_object"}``.
        extra: Arbitrary provider kwargs passthrough.
    """

    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None
    top_logprobs: int | None = None
    seed: int | None = None
    response_format: dict | None = None
    extra: dict | None = None


@dataclass
class ChoiceRequest:
    """Score a fixed option set under a prompt (e.g. ``["Yes", "No"]``).

    Args:
        messages: Prompt messages, same format as :class:`LLMRequest`.
        options: Options to score; each must be a single leading token
            (``"Yes"``/``"No"``/``"A"``-``"D"``).
    """

    messages: list[dict]
    options: list[str]


@dataclass
class TokenLogprob:
    """One candidate for the first generated token.

    Args:
        token: Candidate token text.
        logprob: Natural log of its probability.
    """

    token: str
    logprob: float


@dataclass
class LLMResponse:
    """Result of one generation request.

    Args:
        text: Generated text.
        logprobs: First-token distribution when ``top_logprobs`` was
            requested (None when absent or dropped by the backend).
        prompt_tokens: Input token count.
        completion_tokens: Output token count.
        cost_usd: Estimated request cost; None when unknown (self-hosted
            endpoints have no pricing map).
        cached: Whether this response came from the response cache.
    """

    text: str
    logprobs: list[TokenLogprob] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    cached: bool = False


@dataclass
class LLMUsage:
    """Cumulative client accounting.

    Token counts and cost cover only requests that reached the transport
    (cached responses cost nothing); ``cache_hits`` counts the rest.

    Args:
        requests: Requests sent to the transport (generate + embed).
        prompt_tokens: Input tokens across transported requests.
        completion_tokens: Output tokens across transported requests.
        cache_hits: Requests served from the response cache.
        cost_usd: Estimated total cost.
    """

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0


def _response_payload(resp: LLMResponse) -> dict[str, Any]:
    """Convert a response to its JSON-safe cache payload."""
    return {
        "text": resp.text,
        "logprobs": (
            [asdict(t) for t in resp.logprobs] if resp.logprobs is not None else None
        ),
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "cost_usd": resp.cost_usd,
    }


def _response_from_payload(payload: dict[str, Any]) -> LLMResponse:
    """Rebuild a cached response from its payload."""
    logprobs = payload.get("logprobs")
    return LLMResponse(
        text=payload["text"],
        logprobs=(
            [TokenLogprob(**t) for t in logprobs] if logprobs is not None else None
        ),
        prompt_tokens=payload.get("prompt_tokens", 0),
        completion_tokens=payload.get("completion_tokens", 0),
        cost_usd=payload.get("cost_usd"),
        cached=True,
    )


class LLMClient(ABC):
    """Base class for LLM inference backends.

    ``generate``/``embed`` are cache-aware template methods; subclasses
    implement only the transport hooks ``_complete``/``_embed`` (concurrency
    and provider protocol live there). ``choice`` composes ``generate`` and
    normalizes first-token probability mass over the option set.

    Args:
        cfg: LLM configuration node (model, sampling defaults, cache path).
    """

    def __init__(self, cfg: LLMConfig) -> None:
        """Initialize the client with its cache and usage accounting.

        Args:
            cfg: LLM configuration node (model, sampling defaults, cache path).
        """
        self.cfg = cfg
        self.usage = LLMUsage()
        self._cache = SQLiteResponseCache(cfg.cache_path) if cfg.cache_path else None

    def generate(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Cached batch generation; cache hits skip the transport entirely.

        Args:
            requests: Requests to run, answered in the same order.

        Returns:
            One :class:`LLMResponse` per request.
        """
        if not requests:
            return []
        keys = [self._request_key(r) for r in requests]
        cached = self._cache.get_many(keys) if self._cache else {}
        results: list[LLMResponse | None] = [None] * len(requests)
        misses: list[int] = []
        for i, key in enumerate(keys):
            payload = cached.get(key)
            if payload is not None:
                results[i] = _response_from_payload(payload)
            else:
                misses.append(i)
        if misses:
            fresh = self._complete([requests[i] for i in misses])
            if len(fresh) != len(misses):
                raise LLMError(
                    f"{type(self).__name__}._complete returned {len(fresh)} "
                    f"responses for {len(misses)} requests"
                )
            for pos, resp in zip(misses, fresh, strict=True):
                results[pos] = resp
                self._account(resp)
                if self._cache is not None:
                    self._cache.put(keys[pos], _response_payload(resp))
        for r in results:
            if r is not None and r.cached:
                self.usage.cache_hits += 1
        logger.debug(
            "generate: %d requests (%d cached, %d transported)",
            len(requests),
            len(requests) - len(misses),
            len(misses),
        )
        return cast(list[LLMResponse], results)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Cached batch embeddings; deterministic, hence always cacheable.

        Args:
            texts: Input strings.

        Returns:
            One embedding vector per input, in order.
        """
        if not texts:
            return []
        if not self.cfg.embedding_model:
            raise LLMError(
                "llm.embedding_model is not configured — embed() requires it"
            )
        keys = [self._embed_key(t) for t in texts] if self._cache else []
        cached = self._cache.get_many(keys) if self._cache else {}
        results: list[list[float] | None] = [None] * len(texts)
        miss_pos: list[int] = []
        hits = 0
        for i in range(len(texts)):
            payload = cached.get(keys[i]) if keys else None
            if payload is not None:
                results[i] = payload["vector"]
                hits += 1
            else:
                miss_pos.append(i)
        if miss_pos:
            fresh = self._embed([texts[i] for i in miss_pos])
            if len(fresh) != len(miss_pos):
                raise LLMError(
                    f"{type(self).__name__}._embed returned {len(fresh)} vectors "
                    f"for {len(miss_pos)} inputs"
                )
            for pos, vector in zip(miss_pos, fresh, strict=True):
                results[pos] = vector
            self.usage.requests += len(miss_pos)
            if self._cache is not None:
                for pos, vector in zip(miss_pos, fresh, strict=True):
                    self._cache.put(keys[pos], {"vector": vector})
        self.usage.cache_hits += hits
        return cast(list[list[float]], results)

    def choice(self, requests: list[ChoiceRequest]) -> list[dict[str, float]]:
        """Option probabilities from the first generated token's distribution.

        Each request is scored as one-token generation with ``top_logprobs``;
        probability mass is matched per option (leading-space token variants
        included) and normalized across the option set. Requires a backend
        that returns logprobs (OpenAI-compatible endpoints).

        Args:
            requests: Prompts with their option sets.

        Returns:
            One ``{option: probability}`` dict per request.
        """
        responses = self.generate(
            [
                LLMRequest(messages=r.messages, max_tokens=1, top_logprobs=20)
                for r in requests
            ]
        )
        return [
            self._normalize_choice(r.options, resp)
            for r, resp in zip(requests, responses, strict=True)
        ]

    @abstractmethod
    def _complete(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Transport hook: answer the given (uncached) requests."""

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Transport hook: embed the given (uncached) texts."""

    def _account(self, resp: LLMResponse) -> None:
        """Fold one transported response into the usage counters."""
        self.usage.requests += 1
        self.usage.prompt_tokens += resp.prompt_tokens
        self.usage.completion_tokens += resp.completion_tokens
        self.usage.cost_usd += resp.cost_usd if resp.cost_usd is not None else 0.0

    def _request_key(self, req: LLMRequest) -> str:
        """Cache key for one generate request (resolved params included)."""
        resolved = {
            "messages": req.messages,
            "temperature": self._resolve(req.temperature, self.cfg.temperature),
            "max_tokens": self._resolve(req.max_tokens, self.cfg.max_tokens),
            "top_logprobs": req.top_logprobs,
            "seed": req.seed,
            "response_format": req.response_format,
            "extra": req.extra,
        }
        return cache_key("generate", self.cfg.model, self.cfg.api_base, resolved)

    def _embed_key(self, text: str) -> str:
        """Cache key for one embed input."""
        return cache_key("embed", self.cfg.embedding_model, self.cfg.api_base, text)

    @staticmethod
    def _resolve(value: Any, default: Any) -> Any:
        """Return ``value`` unless None, then ``default``."""
        return value if value is not None else default

    @staticmethod
    def _normalize_choice(options: list[str], resp: LLMResponse) -> dict[str, float]:
        """Normalize first-token probability mass over the option set."""
        if not resp.logprobs:
            raise LLMError(
                "choice() requires logprobs but the response carries none — "
                "the backend dropped them or does not support logprobs"
            )
        weights = {
            opt: sum(
                math.exp(t.logprob)
                for t in resp.logprobs
                if t.token.strip() == opt.strip()
            )
            for opt in options
        }
        total = sum(weights.values())
        if total <= 0.0:
            top = ", ".join(repr(t.token) for t in resp.logprobs[:5])
            raise LLMError(
                f"no probability mass on options {options}; first-token "
                f"candidates were: {top}"
            )
        return {opt: w / total for opt, w in weights.items()}


def create_llm_client(cfg: LLMConfig | None) -> LLMClient:
    """Instantiate the configured client backend (fail fast when unset).

    Args:
        cfg: The ``rc.llm`` node (or a standalone :class:`LLMConfig`).

    Returns:
        A client bound to ``cfg``.

    Raises:
        ValueError: If ``llm.model`` is empty (LLM access not configured).
        KeyError: If ``llm.provider`` names no registered backend (the error
            lists the available providers).
    """
    if cfg is None or not cfg.model:
        raise ValueError(
            "LLM access requested but llm.model is empty — configure it via "
            "--llm.model (a remote model string, or a self-hosted "
            "OpenAI-compatible endpoint together with --llm.api_base)"
        )
    cls = LLM_CLIENTS.get(cfg.provider)
    return cast(LLMClient, cls(cfg))


__all__ = [
    "ChoiceRequest",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "TokenLogprob",
    "create_llm_client",
]
