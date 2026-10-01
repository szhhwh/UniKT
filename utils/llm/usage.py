"""Trainer-side observability for LLM clients.

Imported explicitly by trainers that use an LLM client (this module pulls
the training callback stack, which ``utils.llm`` itself stays free of).
"""

from dataclasses import asdict
from typing import Any

from utils.config import update_run_metadata
from utils.core import get_logger
from utils.training.callbacks import Callback

from .base import LLMClient

logger = get_logger(__name__)


class LLMUsageCallback(Callback):
    """Report :attr:`LLMClient.usage` when a (stage) run ends.

    Fans usage out through the trainer's metric logger (``log_final``:
    metrics_final.csv + SwanLab/W&B) and merges an ``llm_usage`` block into
    ``run_metadata.yaml``. Multi-stage trainers fire ``on_train_end`` per
    stage with the same callback instance, so reports are gated on the usage
    snapshot having changed since the last one.

    Args:
        client: The client whose cumulative usage is reported.
    """

    def __init__(self, client: LLMClient) -> None:
        """Bind the callback to a client and clear the report state.

        Args:
            client: The client whose cumulative usage is reported.
        """
        self.client = client
        self._trainer: Any = None
        self._reported = asdict(client.usage)

    def on_train_begin(self, epochs: int, **kwargs: Any) -> None:
        """Remember the trainer for the crash-path report in ``close``.

        Args:
            epochs: Total number of epochs.
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        trainer = kwargs.get("trainer")
        if trainer is not None:
            self._trainer = trainer

    def on_train_end(self, **kwargs: Any) -> None:
        """Report usage on normal (per-stage) completion.

        Args:
            **kwargs: Additional keyword arguments (e.g. trainer).
        """
        trainer = kwargs.get("trainer") or self._trainer
        self._report(trainer)

    def close(self) -> None:
        """Best-effort report for failed runs (``on_train_end`` never fires).

        The metric logger may already be finished, so any failure here is
        downgraded to a warning; ``close`` must never raise.
        """
        try:
            self._report(self._trainer)
        except Exception as e:
            logger.warning(f"LLM usage reporting failed: {e}")

    def _report(self, trainer: Any) -> None:
        """Emit one report if usage moved since the previous one."""
        snapshot = asdict(self.client.usage)
        if snapshot == self._reported:
            return
        self._reported = snapshot
        metric_logger = getattr(trainer, "metric_logger", None) if trainer else None
        if metric_logger is not None:
            metric_logger.log_final(
                metrics={
                    "LLM/Requests": snapshot["requests"],
                    "LLM/Prompt_Tokens": snapshot["prompt_tokens"],
                    "LLM/Completion_Tokens": snapshot["completion_tokens"],
                    "LLM/Cache_Hits": snapshot["cache_hits"],
                    "LLM/Cost_USD": snapshot["cost_usd"],
                },
                step=getattr(trainer, "_global_step", 0) if trainer else 0,
            )
        log_dir = getattr(trainer, "log_dir", None) if trainer else None
        if log_dir:
            update_run_metadata(log_dir, {"llm_usage": snapshot})
        logger.info(
            "LLM usage: %(requests)d requests (%(cache_hits)d cache hits), "
            "%(prompt_tokens)d+%(completion_tokens)d tokens, $%(cost_usd).4f",
            snapshot,
        )


__all__ = ["LLMUsageCallback"]
