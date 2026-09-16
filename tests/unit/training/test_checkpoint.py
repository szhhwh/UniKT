"""Tests for ``utils.training.checkpoint``: queue coalescing + lifecycle edges.

Focuses on the async-save queue coalescing in ``CheckpointManager``: when the
single background worker cannot keep up (slow disk), successive saves to the
same path must collapse so ``close()`` does not drain a long tail of stale,
already-superseded writes. A ``slow_write`` fixture holds the worker on an
``Event`` so pending state is deterministic rather than racy.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from utils.training.checkpoint import CheckpointManager


class _SlowGate:
    """Release gate + running flag for the ``slow_write`` fixture.

    ``set`` releases the blocked worker; ``wait_running`` blocks until the
    worker has entered the slow section (the first save is running), so a test
    can assert against a running future without a scheduling race.
    """

    def __init__(self) -> None:
        self._release = threading.Event()
        self._running = threading.Event()

    def set(self) -> None:
        self._release.set()

    def wait_running(self, timeout: float = 2.0) -> bool:
        return self._running.wait(timeout)


@pytest.fixture
def slow_write(monkeypatch: pytest.MonkeyPatch) -> Iterator[_SlowGate]:
    """Block the saver worker so submits pile up in the queue.

    Yields a gate: the test releases the worker (``gate.set()``) before
    calling ``close()``, and teardown sets it again as a safety net so a
    blocked worker can never hang the executor shutdown.
    """
    real = CheckpointManager._write_atomic
    gate = _SlowGate()

    def _slow(obj: object, filepath: str) -> None:
        gate._running.set()
        gate._release.wait()
        real(obj, filepath)

    monkeypatch.setattr(CheckpointManager, "_write_atomic", staticmethod(_slow))
    yield gate
    gate.set()


# ---------------------------------------------------------------------------
# coalescing: same-path pending saves are cancelled, running ones are not
# ---------------------------------------------------------------------------


class TestCoalesce:
    def test_same_path_pending_cancelled(
        self, tmp_path: Path, slow_write: _SlowGate
    ) -> None:
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "last_checkpoint.pth")
        for i in range(5):
            mgr._submit_save({"v": i}, path)

        # only the newest save per path is tracked; older pending ones cancelled
        assert len(mgr._latest) == 1

        slow_write.set()
        mgr.close()

    def test_running_save_not_cancelled(
        self, tmp_path: Path, slow_write: _SlowGate
    ) -> None:
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "last_checkpoint.pth")
        mgr._submit_save({"v": 0}, path)  # picked up by the worker, now blocked
        assert slow_write.wait_running()  # worker is inside the first save
        running = mgr._latest[path]
        assert not running.done()

        # a newer same-path submit cannot cancel the running future
        mgr._submit_save({"v": 1}, path)
        assert not running.cancelled()

        slow_write.set()
        mgr.close()

    def test_different_paths_kept(self, tmp_path: Path, slow_write: _SlowGate) -> None:
        mgr = CheckpointManager(str(tmp_path))
        best = str(tmp_path / "best_model.pth")
        last = str(tmp_path / "last_checkpoint.pth")
        mgr._submit_save({"v": 1}, best)
        mgr._submit_save({"v": 1}, last)

        assert best in mgr._latest and last in mgr._latest

        slow_write.set()
        mgr.close()

    def test_newest_value_persisted(
        self, tmp_path: Path, slow_write: _SlowGate
    ) -> None:
        # invariant under any scheduling: the last submit wins on disk,
        # because every older pending save to the same path is cancelled
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "last_checkpoint.pth")
        for i in range(10):
            mgr._submit_save({"v": i}, path)

        slow_write.set()
        mgr.close()

        data = torch.load(path, weights_only=True)
        assert data["v"] == 9


# ---------------------------------------------------------------------------
# lifecycle edges
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_idempotent(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        mgr.close()
        mgr.close()  # second close must be a no-op

    def test_submit_after_close_writes_synchronously(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        mgr.close()

        path = str(tmp_path / "x.pth")
        mgr._submit_save({"v": 1}, path)

        # closed manager bypasses the executor: file exists now, nothing tracked
        assert os.path.exists(path)
        assert mgr._latest == {}

    def test_flush_drains_queue(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "x.pth")
        mgr._submit_save({"v": 1}, path)

        mgr.flush()

        assert mgr._latest == {}
        assert os.path.exists(path)

    def test_flush_empty_is_noop(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        mgr.flush()  # draining an empty queue must not raise

    def test_no_tmp_residue(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "x.pth")
        mgr._submit_save({"v": 1}, path)
        mgr.close()

        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")


# ---------------------------------------------------------------------------
# save_weights: snapshot semantics survive coalescing
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_returns_cpu_snapshot_and_writes(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(3, 2)

        snapshot = mgr.save_weights(model, "best.pth")
        mgr.close()

        assert isinstance(snapshot, dict)
        assert all(t.device.type == "cpu" for t in snapshot.values())
        assert os.path.exists(tmp_path / "best.pth")

    def test_snapshot_returned_every_call_when_coalesced(
        self, tmp_path: Path, slow_write: _SlowGate
    ) -> None:
        # disk writes coalesce, but each call still returns the snapshot taken
        # at that moment — it backs the in-memory best-state cache
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(3, 2)

        snapshots = [mgr.save_weights(model, "best.pth") for _ in range(5)]

        assert all(s is not None for s in snapshots)
        slow_write.set()
        mgr.close()


# ---------------------------------------------------------------------------
# error handling: a failed save must not escape flush/close
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_failed_save_does_not_raise_on_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(obj: object, filepath: str) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr(CheckpointManager, "_write_atomic", staticmethod(boom))

        mgr = CheckpointManager(str(tmp_path))
        mgr._submit_save({"v": 1}, str(tmp_path / "x.pth"))

        # exception is logged inside _reap, never propagated to the caller
        mgr.close()


# ---------------------------------------------------------------------------
# save_checkpoint payload shape
# ---------------------------------------------------------------------------


class TestSavePayload:
    def test_minimal_payload_keys(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr.save_checkpoint(3, model, opt, filename="ckpt.pth")
        mgr.close()

        saved = torch.load(tmp_path / "ckpt.pth", weights_only=False)
        assert saved["epoch"] == 3
        assert "model_state_dict" in saved and "optimizer_state_dict" in saved
        assert "rng_states" in saved
        assert "scheduler_state_dict" not in saved
        assert "early_stopping_state" not in saved

    def test_scheduler_and_es_state_included(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr.save_checkpoint(
            4,
            model,
            opt,
            scheduler=sched,
            early_stopping_state={"best_score": 0.9},
            filename="ckpt.pth",
        )
        mgr.close()

        saved = torch.load(tmp_path / "ckpt.pth", weights_only=False)
        assert "scheduler_state_dict" in saved
        assert saved["early_stopping_state"] == {"best_score": 0.9}

    def test_additional_state_overrides_reserved_keys(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr.save_checkpoint(
            3,
            model,
            opt,
            additional_state={"epoch": 99, "custom": "x"},
            filename="ckpt.pth",
        )
        mgr.close()

        saved = torch.load(tmp_path / "ckpt.pth", weights_only=False)
        assert saved["epoch"] == 99  # additional_state wins over reserved keys
        assert saved["custom"] == "x"


# ---------------------------------------------------------------------------
# _detach_to_cpu recursion
# ---------------------------------------------------------------------------


class TestDetachToCpu:
    def test_nested_structures_recursed(self) -> None:
        from utils.training.checkpoint import _detach_to_cpu

        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        out = _detach_to_cpu({"a": [leaf, (leaf + 1,)], "b": {"c": "plain"}, "d": 5})
        assert isinstance(out["a"], list)
        assert isinstance(out["a"][1], tuple)
        assert out["b"]["c"] == "plain"
        assert out["d"] == 5
        flat = out["a"][0]
        assert not flat.requires_grad
        assert flat.device.type == "cpu"
        assert torch.equal(flat.detach(), leaf)

    def test_tensor_clone_is_independent(self) -> None:
        from utils.training.checkpoint import _detach_to_cpu

        original = torch.tensor([1.0])
        cloned = _detach_to_cpu(original)
        cloned.mul_(10)
        assert original.item() == 1.0

    def test_non_tensor_leaf_untouched(self) -> None:
        from utils.training.checkpoint import _detach_to_cpu

        assert _detach_to_cpu("str") == "str"
        assert _detach_to_cpu(None) is None


# ---------------------------------------------------------------------------
# RNG capture / restore round-trip
# ---------------------------------------------------------------------------


class TestRngStates:
    def test_capture_restore_round_trip_torch_numpy_python(self) -> None:
        import random as random_module

        import numpy as np

        from utils.training.checkpoint import _capture_rng_states, _restore_rng_states

        torch.manual_seed(123)
        np.random.seed(123)
        random_module.seed(123)
        states = _capture_rng_states()

        # Advance every RNG so the current state differs from the snapshot.
        torch.rand(3)
        np.random.rand(3)
        random_module.random()

        _restore_rng_states(states)
        t1 = torch.rand(3)
        n1 = np.random.rand(3)
        p1 = random_module.random()

        _restore_rng_states(states)
        assert torch.equal(torch.rand(3), t1)
        assert np.array_equal(np.random.rand(3), n1)
        assert random_module.random() == p1

    def test_captured_numpy_state_is_weights_only_safe(self) -> None:
        from utils.training.checkpoint import _capture_rng_states

        states = _capture_rng_states()
        np_key = states["numpy"]
        assert isinstance(np_key[1], list)  # not an ndarray

    def test_cuda_state_none_on_cpu(self) -> None:
        from utils.training.checkpoint import _capture_rng_states

        if torch.cuda.is_available():
            pytest.skip("CUDA present; the None branch is not reachable")
        assert _capture_rng_states()["cuda"] is None


# ---------------------------------------------------------------------------
# torch.compile prefix stripping
# ---------------------------------------------------------------------------


class TestCompilePrefix:
    def test_strips_prefix_into_raw_model(self) -> None:
        from utils.training.checkpoint import _strip_compile_prefix_if_needed

        model = torch.nn.Linear(2, 1)
        prefixed = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
        stripped = _strip_compile_prefix_if_needed(prefixed, model)
        assert set(stripped) == set(model.state_dict())

    def test_keeps_prefix_when_model_is_compiled(self) -> None:
        from utils.training.checkpoint import _strip_compile_prefix_if_needed

        model = torch.nn.Linear(2, 1)
        prefixed = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
        # A model whose own keys are prefixed (compiled) matches the state as-is.
        fake_compiled = type("M", (), {"state_dict": lambda self: prefixed})()
        assert _strip_compile_prefix_if_needed(prefixed, fake_compiled) is prefixed

    def test_plain_state_untouched(self) -> None:
        from utils.training.checkpoint import _strip_compile_prefix_if_needed

        model = torch.nn.Linear(2, 1)
        plain = model.state_dict()
        assert _strip_compile_prefix_if_needed(plain, model) is plain


# ---------------------------------------------------------------------------
# load_weights / read_model_state_dict / load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadWeights:
    def test_plain_state_dict_returns_none_and_loads(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        src = torch.nn.Linear(2, 1)
        with torch.no_grad():
            src.weight.fill_(7.0)
        mgr.save_weights(src, "plain.pth")
        mgr.close()

        dst = torch.nn.Linear(2, 1)
        result = CheckpointManager.load_weights(str(tmp_path / "plain.pth"), dst)
        assert result is None
        assert torch.equal(dst.weight, src.weight)

    def test_full_checkpoint_returns_dict(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        src = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(src.parameters(), lr=0.1)
        mgr.save_checkpoint(9, src, opt, filename="full.pth")
        mgr.close()

        dst = torch.nn.Linear(2, 1)
        # A full checkpoint payload (vs plain weights) is guaranteed here.
        raw = cast(
            "dict[str, Any]",
            CheckpointManager.load_weights(str(tmp_path / "full.pth"), dst),
        )
        assert raw["epoch"] == 9
        assert torch.equal(dst.weight, src.weight)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            CheckpointManager.load_weights(
                str(tmp_path / "nope.pth"), torch.nn.Linear(2, 1)
            )

    def test_read_model_state_dict_both_formats(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        mgr.save_weights(model, "plain.pth")
        mgr.save_checkpoint(1, model, opt, filename="full.pth")
        mgr.close()

        plain = CheckpointManager.read_model_state_dict(str(tmp_path / "plain.pth"))
        full = CheckpointManager.read_model_state_dict(str(tmp_path / "full.pth"))
        assert set(plain) == set(full) == set(model.state_dict())


class TestLoadCheckpointE2E:
    def test_full_restore_model_optimizer_es(self, tmp_path: Path) -> None:
        from utils.config.run_config import EarlyStoppingConfig
        from utils.training.early_stopping import EarlyStopping

        mgr = CheckpointManager(str(tmp_path))
        src = torch.nn.Linear(2, 1)
        opt = torch.optim.Adam(src.parameters(), lr=0.1)
        # Give the optimizer real per-param state (exp_avg / exp_avg_sq buffers).
        loss = src(torch.ones(1, 2)).sum()
        loss.backward()
        opt.step()

        es = EarlyStopping(EarlyStoppingConfig())
        es.step(0.75, epoch=2, metrics={"auc": 0.75})

        mgr.save_checkpoint(
            2,
            src,
            opt,
            early_stopping_state={
                "best_score": es.best_score,
                "best_epoch": es.best_epoch,
                "num_bad_epochs": es.num_bad_epochs,
                "best_metrics": es.best_metrics,
            },
            filename="ckpt.pth",
        )
        mgr.close()

        dst = torch.nn.Linear(2, 1)
        dst_opt = torch.optim.Adam(dst.parameters(), lr=0.1)
        dst_es = EarlyStopping(EarlyStoppingConfig())
        loaded = mgr.load_checkpoint(
            str(tmp_path / "ckpt.pth"), dst, optimizer=dst_opt, early_stopping=dst_es
        )

        assert loaded["epoch"] == 2
        assert torch.equal(dst.weight, src.weight)
        assert dst_es.best_score == 0.75
        assert dst_es.best_epoch == 2
        assert dst_es.num_bad_epochs == 0
        assert dst_es.best_metrics == {"auc": 0.75}
        # optimizer state actually restored (has per-param state)
        assert len(dst_opt.state) == len(opt.state) and len(dst_opt.state) > 0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            mgr.load_checkpoint(str(tmp_path / "nope.pth"), torch.nn.Linear(2, 1))
