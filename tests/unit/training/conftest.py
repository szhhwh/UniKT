"""Shared fixtures for training-area tests: tiny trainers and stub manager.

Doubles follow the area style: duck-typed inline classes, no ``unittest.mock``.
``Loaders'' are plain lists of ``(x, y)`` tensor tuples — not
``torch.utils.data.Dataset`` instances — so ``BaseTrainer._setup_data_loaders``
passes them through verbatim and no DataLoader workers are spawned. All
filesystem effects stay under ``tmp_path`` via ``make_exp_manager``.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from utils.training.base_trainer import BaseTrainer, StageResult
from utils.training.multi_trainer import MultiTrainer, StageComponents, StageConfig
from utils.training.runtime_components import RuntimeComponents

if TYPE_CHECKING:
    from utils.config.run_config import EarlyStoppingConfig, RunConfig
    from utils.training.callbacks import Callback

_UNSET = object()


class StubExpManager:
    """Duck-typed ExperimentManager: only ``get_log_dir`` + ``is_existing_run`` are read."""

    def __init__(self, log_dir: Path, is_existing_run: bool = False) -> None:
        self.log_dir = str(log_dir)
        self.is_existing_run = is_existing_run

    def get_log_dir(self) -> str:
        return self.log_dir


@pytest.fixture
def make_exp_manager(tmp_path: Path) -> Callable[..., StubExpManager]:
    """Factory: StubExpManager whose log dir is created under tmp_path."""

    def _make(
        log_dir: str | None = None, is_existing_run: bool = False
    ) -> StubExpManager:
        path = tmp_path / (log_dir or "run0")
        path.mkdir(parents=True, exist_ok=True)
        return StubExpManager(path, is_existing_run=is_existing_run)

    return _make


@pytest.fixture
def make_batches() -> Callable[..., list[tuple[torch.Tensor, torch.Tensor]]]:
    """Factory: deterministic (x, y) tuple batches for tiny linear regression.

    Labels are binary (y == x[:, 0]) so val metrics (auc/acc) are defined.
    """

    def _make(
        batch_sizes: tuple[int, ...] = (4, 2),
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for size in batch_sizes:
            idx = torch.arange(offset, offset + size)
            x = torch.stack([idx % 2, (idx + 1) % 2], dim=1).float()
            y = (idx % 2).float()
            batches.append((x, y))
            offset += size
        return batches

    return _make


def _tiny_outputs(
    model: torch.nn.Module, batch_data: tuple[torch.Tensor, torch.Tensor]
) -> dict[str, Any]:
    """Shared forward for the tiny trainers: linear model over (x, y) tuples."""
    x, y = batch_data
    y_hat = model(x).reshape(-1)
    return {
        "y_hat": y_hat,
        "y_label": y.float(),
        "y_predict": (y_hat >= 0.5).to(torch.int),
        "y_score": y_hat,
        "y_prob": torch.sigmoid(y_hat),
    }


class TinyTrainer(BaseTrainer):
    """Minimal concrete single-stage trainer: Linear(2, 1) + MSELoss + SGD."""

    def __init__(
        self,
        rc: RunConfig | None,
        exp_manager: StubExpManager | None,
        train: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        val: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        test: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        extra_callbacks: Iterable[Callback] = (),
        max_clip_grad_norm: float | None = None,
        lr_scheduler: Any = None,
    ) -> None:
        # Stash before super().__init__ — build_components reads these.
        self._tiny: dict[str, Any] = {
            "train": train,
            "val": val,
            "test": test,
            "extra": list(extra_callbacks),
            "clip": max_clip_grad_norm,
            "sched": lr_scheduler,
        }
        super().__init__(rc, None, exp_manager)

    def build_components(self, rc: Any, data_src: Any) -> RuntimeComponents:
        torch.manual_seed(7)
        model = torch.nn.Linear(2, 1)
        return RuntimeComponents(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=torch.nn.MSELoss(),
            lr_scheduler=self._tiny["sched"],
            train_data=self._tiny["train"],
            val_data=self._tiny["val"],
            test_data=self._tiny["test"],
            max_clip_grad_norm=self._tiny["clip"],
        )

    def build_callbacks(self) -> list[Callback]:
        return list(self._tiny["extra"])

    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        return _tiny_outputs(self.model, batch_data)


@pytest.fixture
def make_tiny_trainer(
    make_run_config: Callable[..., RunConfig],
    make_exp_manager: Callable[..., StubExpManager],
    make_batches: Callable[..., list[tuple[torch.Tensor, torch.Tensor]]],
) -> Callable[..., TinyTrainer]:
    """Factory: a built TinyTrainer on CPU with overridable rc/exp/data/callbacks."""

    def _make(
        *,
        rc: RunConfig | None = None,
        exp: StubExpManager | None = None,
        train: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        val: Any = _UNSET,  # sentinel: None means "explicitly no val data"
        test: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        extra_callbacks: Iterable[Callback] = (),
        max_clip_grad_norm: float | None = None,
        lr_scheduler: Any = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> TinyTrainer:
        rc = rc or make_run_config(model_kwargs={"epochs": 1, **(model_kwargs or {})})
        exp = exp or make_exp_manager()
        return TinyTrainer(
            rc,
            exp,
            make_batches() if train is None else train,
            val=make_batches() if val is _UNSET else val,
            test=test,
            extra_callbacks=extra_callbacks,
            max_clip_grad_norm=max_clip_grad_norm,
            lr_scheduler=lr_scheduler,
        )

    return _make


class TinyMultiTrainer(MultiTrainer):
    """Two tiny stages exercising build_stages / _apply_stage / on_stage_*.

    ``stage_specs`` maps stage name to builder options (``epochs``,
    ``early_stopping``, ``checkpoint_monitor`` / ``checkpoint_mode``); every
    lifecycle event is appended to ``events`` so tests can assert ordering.
    """

    def __init__(
        self,
        rc: RunConfig,
        exp_manager: StubExpManager,
        train: list[tuple[torch.Tensor, torch.Tensor]],
        val: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        stage_specs: dict[str, dict[str, Any]] | None = None,
        extra_callbacks: Iterable[Callback] = (),
    ) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.stage_snapshots: dict[str, dict[str, Any]] = {}
        self._extra = list(extra_callbacks)
        self._specs: dict[str, dict[str, Any]] = stage_specs or {
            "km": {"epochs": 1, "early_stopping": _default_es()},
            "am": {"epochs": 1, "early_stopping": _default_es()},
        }
        self._train = train
        self._val = val
        super().__init__(rc, None, exp_manager)

    def build_stages(self) -> list[StageConfig]:
        return [StageConfig(name, self._make_builder(name)) for name in self._specs]

    def build_callbacks(self) -> list[Callback]:
        return list(self._extra)

    def _make_builder(self, name: str) -> Callable[[], StageComponents]:
        def _build() -> StageComponents:
            self.events.append(("build", name))
            spec: dict[str, Any] = self._specs[name]
            torch.manual_seed(7)
            model = torch.nn.Linear(2, 1)
            return StageComponents(
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                loss_fn=torch.nn.MSELoss(),
                train_data=self._train,
                val_data=self._val,
                epochs=spec.get("epochs", 1),
                early_stopping=spec.get("early_stopping"),
                checkpoint_monitor=spec.get("checkpoint_monitor"),
                checkpoint_mode=spec.get("checkpoint_mode"),
            )

        return _build

    def on_stage_begin(self, name: str) -> None:
        self.events.append(("begin", name))

    def on_stage_complete(self, name: str, result: StageResult) -> None:
        from utils.training.callbacks import (
            CallbackManager,
            CheckpointCallback,
            EarlyStoppingCallback,
        )

        self.events.append(("complete", name, result))
        manager = cast(CallbackManager, self.callback_manager)
        checkpoint_cb = cast(
            CheckpointCallback, manager.get_callback(CheckpointCallback)
        )
        self.stage_snapshots[name] = {
            "has_es": self.early_stopping is not None,
            "has_es_cb": manager.get_callback(EarlyStoppingCallback) is not None,
            "metric_step_offset": self._metric_step_offset,
            "callback_manager": manager,
            "best_filename": checkpoint_cb.best_filename,
            "monitor_override": checkpoint_cb._monitor_override,
            "mode_override": checkpoint_cb._mode_override,
            "es_monitor": self.early_stopping.cfg.monitor
            if self.early_stopping
            else None,
        }

    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        return _tiny_outputs(self.model, batch_data)


def _default_es() -> EarlyStoppingConfig:
    from utils.config.run_config import EarlyStoppingConfig

    return EarlyStoppingConfig(patience=2)


class FakeCloudBackend:
    """Recording stand-in for a swanlab/wandb SDK module."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def login(self, **kwargs: Any) -> None:
        self.calls.append(("login", kwargs))

    def Settings(self) -> object:
        self.calls.append(("settings",))
        return object()

    def init(self, **kwargs: Any) -> None:
        self.calls.append(("init", kwargs))

    def log(self, data: dict[str, Any], **kwargs: Any) -> None:
        self.calls.append(("log", dict(data), kwargs))

    def finish(self) -> None:
        self.calls.append(("finish",))


@pytest.fixture
def inject_fake_cloud_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], FakeCloudBackend]:
    """Install a recording fake ``swanlab``/``wandb`` into sys.modules.

    Both cloud backends lazy-import their SDK inside every method, so a
    ``sys.modules`` entry is enough to intercept them. Returns the recording
    backend; ``monkeypatch.setitem`` restores the real modules on teardown.
    """

    def _install(name: str) -> FakeCloudBackend:
        backend = FakeCloudBackend()
        mod = types.ModuleType(name)
        if name == "swanlab":
            setattr(mod, "login", backend.login)
            setattr(mod, "init", backend.init)
            setattr(mod, "Settings", backend.Settings)
            setattr(mod, "log", backend.log)
            setattr(mod, "finish", backend.finish)

            exceptions = types.ModuleType(f"{name}.exceptions")

            class AuthenticationError(Exception):
                pass

            setattr(exceptions, "AuthenticationError", AuthenticationError)
            setattr(mod, "exceptions", exceptions)

            notification = types.ModuleType(f"{name}.plugin.notification")

            class LarkCallback:
                def __init__(self, **kwargs: Any) -> None:
                    pass

            setattr(notification, "LarkCallback", LarkCallback)
            plugin = types.ModuleType(f"{name}.plugin")
            setattr(plugin, "notification", notification)
            setattr(mod, "plugin", plugin)

            monkeypatch.setitem(sys.modules, name, mod)
            monkeypatch.setitem(sys.modules, f"{name}.exceptions", exceptions)
            monkeypatch.setitem(sys.modules, f"{name}.plugin", plugin)
            monkeypatch.setitem(
                sys.modules, f"{name}.plugin.notification", notification
            )
        else:
            setattr(mod, "init", backend.init)
            setattr(mod, "log", backend.log)
            setattr(mod, "finish", backend.finish)
            monkeypatch.setitem(sys.modules, name, mod)
        return backend

    return _install
