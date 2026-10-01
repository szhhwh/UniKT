"""Tests for LLMUsageCallback reporting (metric logger + run metadata)."""

from pathlib import Path
from typing import Any

from utils.config import LLMConfig, load_run_metadata
from utils.llm import LLMRequest, create_llm_client
from utils.llm.usage import LLMUsageCallback


class _RecordingLogger:
    """Duck-typed metric logger double recording log_final calls."""

    def __init__(self) -> None:
        self.final_calls: list[dict[str, float]] = []
        self.steps: list[int] = []

    def log_final(self, *, metrics: dict[str, float], step: int) -> None:
        self.final_calls.append(metrics)
        self.steps.append(step)


class _StubTrainer:
    """Trainer double exposing exactly what the callback reads."""

    def __init__(self, log_dir: Path, step: int = 7) -> None:
        self.metric_logger: Any = _RecordingLogger()
        self.log_dir = str(log_dir)
        self._global_step = step


def _make_client(tmp_path: Path) -> Any:
    cfg = LLMConfig(
        provider="mock",
        model="mock-model",
        cache_path=str(tmp_path / "cache.sqlite"),
    )
    return create_llm_client(cfg)


class TestLLMUsageCallback:
    def test_reports_on_train_end(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        trainer = _StubTrainer(tmp_path)
        callback = LLMUsageCallback(client)
        callback.on_train_begin(epochs=1, trainer=trainer)

        client.generate([LLMRequest(messages=[{"role": "user", "content": "hi"}])])
        callback.on_train_end(trainer=trainer)

        recorder = trainer.metric_logger
        assert len(recorder.final_calls) == 1
        metrics = recorder.final_calls[0]
        assert metrics["LLM/Requests"] == 1
        assert metrics["LLM/Cache_Hits"] == 0
        assert recorder.steps == [7]
        metadata = load_run_metadata(trainer.log_dir)
        assert metadata["llm_usage"]["requests"] == 1

    def test_zero_usage_reports_nothing(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        trainer = _StubTrainer(tmp_path)
        callback = LLMUsageCallback(client)

        callback.on_train_end(trainer=trainer)

        assert trainer.metric_logger.final_calls == []
        assert load_run_metadata(trainer.log_dir) == {}

    def test_unchanged_usage_skips_duplicate_report(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        trainer = _StubTrainer(tmp_path)
        callback = LLMUsageCallback(client)

        client.generate([LLMRequest(messages=[{"role": "user", "content": "a"}])])
        callback.on_train_end(trainer=trainer)
        callback.on_train_end(trainer=trainer)

        assert len(trainer.metric_logger.final_calls) == 1

    def test_new_usage_reports_again(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        trainer = _StubTrainer(tmp_path)
        callback = LLMUsageCallback(client)

        callback.on_train_end(trainer=trainer)
        client.generate([LLMRequest(messages=[{"role": "user", "content": "a"}])])
        callback.on_train_end(trainer=trainer)

        calls = trainer.metric_logger.final_calls
        assert len(calls) == 1
        assert calls[0]["LLM/Requests"] == 1

    def test_close_reports_after_crash_using_stored_trainer(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        trainer = _StubTrainer(tmp_path)
        callback = LLMUsageCallback(client)
        callback.on_train_begin(epochs=1, trainer=trainer)

        client.generate([LLMRequest(messages=[{"role": "user", "content": "hi"}])])
        # No on_train_end (simulated crash); close() reports best-effort.
        callback.close()

        assert len(trainer.metric_logger.final_calls) == 1
        assert load_run_metadata(trainer.log_dir)["llm_usage"]["requests"] == 1

    def test_close_without_trainer_never_raises(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        callback = LLMUsageCallback(client)
        callback.close()
        assert client.usage.requests == 0
