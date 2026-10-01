"""LiteLLM-backed production client: every provider is a model string.

Provider routing, retries, and endpoint protocols are litellm's job; this
class only adapts its responses to the :class:`~utils.llm.base.LLMClient`
contract and fans requests out with a bounded thread pool.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import litellm

from utils.config.run_config import LLMConfig
from utils.core import get_logger, register_llm_client

from .base import LLMClient, LLMRequest, LLMResponse, TokenLogprob

logger = get_logger(__name__)

# litellm prints unsolicited stdout banners and WARNING logs; silence both
# before the first call so our own logger stays the single reporting channel.
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# OpenAI accepts at most 2048 embedding inputs per request; stay well under.
_EMBED_BATCH = 128


def _get(obj: Any, name: str) -> Any:
    """Read ``name`` from a dict or an attribute-style object."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _parse_top_logprobs(logprobs: Any) -> list[TokenLogprob] | None:
    """Extract the first generated token's distribution from a choice.

    Tolerates the logprobs object arriving as a dict or as an attribute
    object, and the distribution sitting under ``content[0].top_logprobs``
    (OpenAI chat shape) or directly on ``top_logprobs`` (single-token
    generation on some backends). Returns None when nothing usable is found.
    """
    if not logprobs:
        return None
    top = None
    content = _get(logprobs, "content")
    if content:
        top = _get(content[0], "top_logprobs")
    if not top:
        top = _get(logprobs, "top_logprobs")
    if not top:
        return None
    out: list[TokenLogprob] = []
    for entry in top:
        token = _get(entry, "token")
        logprob = _get(entry, "logprob")
        if token is None or logprob is None:
            continue
        out.append(TokenLogprob(token=str(token), logprob=float(logprob)))
    return out or None


@register_llm_client("litellm")
class LiteLLMClient(LLMClient):
    """Transport backed by ``litellm.completion``/``litellm.embedding``.

    The provider is chosen by the LiteLLM model string (``"gpt-4o-mini"``,
    ``"anthropic/claude-..."``, ``"hosted_vllm/<model>"`` + ``api_base``,
    ...); retries and timeouts are delegated to litellm via ``num_retries``
    and ``timeout``.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        """Resolve the API key at construction time (``.env`` is loaded by then).

        Args:
            cfg: LLM configuration node.
        """
        super().__init__(cfg)
        self._api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> str | None:
        """Return the API key, or None to pass no key to litellm.

        Raises:
            ValueError: If ``api_key_env`` names an unset/empty variable.
        """
        if not self.cfg.api_key_env:
            return None
        key = os.environ.get(self.cfg.api_key_env, "").strip()
        if not key:
            raise ValueError(
                f"llm.api_key_env is set to '{self.cfg.api_key_env}' but that "
                "variable is empty or unset"
            )
        return key

    def _complete(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Fan the requests out to litellm with a bounded thread pool."""
        if len(requests) <= 1 or self.cfg.max_concurrency <= 1:
            return [self._run_one(r) for r in requests]
        with ThreadPoolExecutor(max_workers=self.cfg.max_concurrency) as pool:
            return list(pool.map(self._run_one, requests))

    def _run_one(self, req: LLMRequest) -> LLMResponse:
        """Send one request through litellm and adapt the response."""
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": req.messages,
            "temperature": (
                req.temperature if req.temperature is not None else self.cfg.temperature
            ),
            "max_tokens": req.max_tokens
            if req.max_tokens is not None
            else self.cfg.max_tokens,
            "num_retries": self.cfg.max_retries,
            "timeout": self.cfg.timeout_s,
        }
        if req.seed is not None:
            kwargs["seed"] = req.seed
        if req.response_format is not None:
            kwargs["response_format"] = req.response_format
        if req.top_logprobs is not None:
            # logprobs must travel together with top_logprobs; alone it
            # yields a null distribution on OpenAI-compatible backends.
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = req.top_logprobs
        if req.extra:
            kwargs.update(req.extra)
        if self.cfg.api_base:
            kwargs["api_base"] = self.cfg.api_base
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        response = litellm.completion(**kwargs)
        return self._to_response(response)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in provider-sized chunks, preserving input order."""
        kwargs: dict[str, Any] = {
            "model": self.cfg.embedding_model,
            "timeout": self.cfg.timeout_s,
        }
        if self.cfg.api_base:
            kwargs["api_base"] = self.cfg.api_base
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            chunk = texts[start : start + _EMBED_BATCH]
            response = litellm.embedding(input=chunk, **kwargs)
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append([float(x) for x in item.embedding])
        return vectors

    @staticmethod
    def _to_response(response: Any) -> LLMResponse:
        """Adapt a litellm chat response to :class:`LLMResponse`."""
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        # Local endpoints have no pricing map: _hidden_params carries None
        # there, while completion_cost() would raise — read the dict instead.
        cost = getattr(response, "_hidden_params", {}).get("response_cost")
        return LLMResponse(
            text=choice.message.content or "",
            logprobs=_parse_top_logprobs(getattr(choice, "logprobs", None)),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=float(cost) if cost is not None else None,
        )


__all__ = ["LiteLLMClient"]
