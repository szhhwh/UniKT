"""Tests for batch helpers: shape probing, recursive device moves, scored counting.

``count_valid_interactions`` / ``count_test_predictions`` are exercised through
the duck-typed ``_StubTarget`` double (pattern shared with
``test_windowlate_stage.py``) so no trainer is built.
"""

from typing import Any, cast

import torch

from utils.efficiency.measures.batch import (
    batch_size_of,
    count_test_predictions,
    count_valid_interactions,
    count_valid_interactions_split,
    to_device,
)
from utils.efficiency.target import BenchmarkTarget


class _StubTarget:
    """Duck-typed BenchmarkTarget counting forward/test_forward calls."""

    def __init__(self, train_y: torch.Tensor, test_y: torch.Tensor) -> None:
        self.model = torch.nn.Linear(2, 2)
        self.device = torch.device("cpu")
        self.forward_calls = 0
        self.test_forward_calls = 0
        self._train_y = train_y
        self._test_y = test_y

    # ``batch`` is an opaque passthrough token; callers hand it back untouched.
    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        return {"y_label": self._train_y}

    def test_forward(self, batch: Any) -> dict[str, torch.Tensor]:
        self.test_forward_calls += 1
        return {"y_label": self._test_y}


# --- batch_size_of ---


class TestBatchSizeOf:
    def test_tuple_of_tensors_uses_first(self) -> None:
        batch = (torch.zeros(4, 3), torch.zeros(4))
        assert batch_size_of(batch) == 4

    def test_list_and_dict_batches(self) -> None:
        assert batch_size_of([torch.zeros(7, 2)]) == 7
        assert batch_size_of({"a": torch.zeros(9, 1)}) == 9

    def test_no_tensor_returns_zero(self) -> None:
        assert batch_size_of(("meta", 42)) == 0
        assert batch_size_of(None) == 0

    def test_nested_list_of_tensors_not_descended(self) -> None:
        # Depth-1 search only: the tensor inside the inner list is invisible.
        assert batch_size_of([[torch.zeros(5, 2)]]) == 0

    def test_scalar_tensor_returns_zero(self) -> None:
        assert batch_size_of(torch.tensor(3)) == 0


# --- to_device ---


class TestToDevice:
    def test_tuple_and_list_types_preserved(self) -> None:
        batch = (torch.zeros(2), torch.zeros(2))
        moved = to_device(batch, torch.device("cpu"))
        assert isinstance(moved, tuple)
        assert isinstance(to_device(list(batch), torch.device("cpu")), list)

    def test_dict_keys_and_values_preserved(self) -> None:
        batch = {"x": torch.zeros(2), "mask": torch.ones(2, dtype=torch.bool)}
        moved = to_device(batch, torch.device("cpu"))
        assert set(moved) == {"x", "mask"}
        assert moved["mask"].dtype == torch.bool

    def test_nested_structure_recurses(self) -> None:
        batch = {"pairs": [torch.zeros(2), (torch.zeros(2), "tag")]}
        moved = to_device(batch, torch.device("cpu"))
        assert isinstance(moved["pairs"][1], tuple)
        assert moved["pairs"][1][1] == "tag"

    def test_non_tensor_passes_through_unchanged(self) -> None:
        assert to_device("meta", torch.device("cpu")) == "meta"
        assert to_device(3, torch.device("cpu")) == 3


# --- counted forwards via the stub target ---


class TestCountScored:
    # The stubs are partial doubles (only forward/device are exercised), so
    # each call casts to the typed BenchmarkTarget boundary.
    def test_valid_interactions_counts_y_label_numel(self) -> None:
        target = _StubTarget(torch.zeros(6), torch.zeros(2))
        assert count_valid_interactions(cast(BenchmarkTarget, target), "batch") == 6
        assert target.forward_calls == 1

    def test_test_predictions_counts_y_label_numel(self) -> None:
        target = _StubTarget(torch.zeros(6), torch.zeros(3, 2))
        assert count_test_predictions(cast(BenchmarkTarget, target), "batch") == 6
        assert target.test_forward_calls == 1

    def test_train_and_test_denominators_are_independent(self) -> None:
        target = _StubTarget(torch.zeros(64), torch.zeros(8))
        assert count_valid_interactions(cast(BenchmarkTarget, target), "b") == 64
        assert count_test_predictions(cast(BenchmarkTarget, target), "b") == 8


class _SplitStubTarget:
    """Duck-typed target whose per-batch ``y_label`` size is batch-dependent."""

    def __init__(self, sizes: list[int]) -> None:
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(1, 1)
        self._sizes = sizes
        self.forward_calls = 0

    def forward(self, batch: int) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        return {"y_label": torch.zeros(self._sizes[batch])}


class TestCountValidInteractionsSplit:
    def test_totals_and_batch_count_over_loader(self) -> None:
        target = _SplitStubTarget([10, 20, 30])
        total, batches = count_valid_interactions_split(
            cast(BenchmarkTarget, target), [0, 1, 2]
        )
        assert total == 60
        assert batches == 3
        assert target.forward_calls == 3

    def test_total_is_invariant_to_batch_order(self) -> None:
        # Shuffle-order invariance is the point of the split count: reordering
        # the loader changes which batch is first, never the sum.
        assert count_valid_interactions_split(
            cast(BenchmarkTarget, _SplitStubTarget([10, 20, 30])), [2, 0, 1]
        ) == (
            60,
            3,
        )

    def test_empty_loader_returns_zero_total_and_count(self) -> None:
        assert count_valid_interactions_split(
            cast(BenchmarkTarget, _SplitStubTarget([])), []
        ) == (0, 0)


class _LazyCacheTarget:
    """Duck-typed target around a module that caches a constant on first forward.

    The cached tensor feeds a grad-tracked multiplication, reproducing the
    AKT-family lazy-constant pattern (e.g. FlucKT's Kerple bias).
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.cached: torch.Tensor | None = None

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        x = batch
        if self.cached is None:
            self.cached = torch.arange(3, dtype=x.dtype, device=x.device)
        return {"y_label": ((x * self.weight) * self.cached).reshape(-1)}


class TestCountScoredCacheSafety:
    def test_counting_forward_keeps_lazy_caches_autograd_compatible(self) -> None:
        """The setup counting forward must not create inference-tensor caches.

        It is usually the session's first forward, so lazily built constants
        are cached here and reused by the grad-enabled FLOPs/train stages;
        inference tensors would be rejected there.
        """
        target = _LazyCacheTarget()
        batch = torch.randn(2, 3)
        assert count_valid_interactions(cast(BenchmarkTarget, target), batch) == 6
        assert target.cached is not None
        assert not target.cached.is_inference()

        out = target.forward(batch)
        out["y_label"].sum().backward()
        assert target.weight.grad is not None
