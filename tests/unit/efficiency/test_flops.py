"""Tests for FLOPs/disk-size estimation: element-size math, CPU FlopCounterMode."""

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from utils.efficiency.measures.flops import (
    count_flops,
    estimate_disk_size_mb,
    format_breakdown,
)


class _FakeFlopCounter(FlopCounterMode):
    """Stand-in counter: only ``get_flop_counts`` is faked, matching the
    ``format_breakdown`` contract; the real mode's dispatch machinery is
    never exercised here."""

    def __init__(
        self,
        counts: dict[str, dict[str, int]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._counts = counts if counts is not None else {}
        self._error = error

    def get_flop_counts(self) -> dict[str, dict[str, int]]:
        if self._error is not None:
            raise self._error
        return self._counts


class TestEstimateDiskSizeMb:
    def test_params_and_buffers_by_element_size(self) -> None:
        model = torch.nn.Linear(3, 2)  # 3*2 weights + 2 bias = 8 params
        model.register_buffer("stats", torch.zeros(4, dtype=torch.float32))
        expected = (8 + 4) * 4 / 1024**2
        assert estimate_disk_size_mb(model) == pytest.approx(expected)

    def test_dtype_doubles_bytes(self) -> None:
        f32 = torch.nn.Linear(4, 4).float()
        f64 = torch.nn.Linear(4, 4).double()
        assert estimate_disk_size_mb(f64) == pytest.approx(
            2 * estimate_disk_size_mb(f32)
        )


class TestCountFlops:
    def test_cpu_linear_forward_counted(self) -> None:
        model = torch.nn.Linear(3, 2)
        flops, breakdown = count_flops(
            lambda: model(torch.randn(4, 3)), torch.device("cpu")
        )
        # addmm: 2 * m * n * k = 2 * 4 * 2 * 3
        assert flops == 48
        assert breakdown.get("addmm") == 48

    def test_exception_returns_none_and_empty(self) -> None:
        def boom() -> None:
            raise RuntimeError("kernel gone")

        flops, breakdown = count_flops(boom, torch.device("cpu"))
        assert flops is None
        assert breakdown == {}


class TestFormatBreakdown:
    def test_op_names_shortened_and_sorted_desc(self) -> None:
        counter = _FakeFlopCounter(
            {"Global": {"aten.addmm": 30, "aten.mm": 10, "aten.sigmoid": 20}}
        )
        assert format_breakdown(counter) == {
            "addmm": 30,
            "sigmoid": 20,
            "mm": 10,
        }

    def test_colon_op_name_unchanged(self) -> None:
        # Only "." is shortened: names like "aten::add" pass through whole.
        counter = _FakeFlopCounter({"Global": {"aten::add": 5}})
        assert format_breakdown(counter) == {"aten::add": 5}

    def test_exception_returns_empty_dict(self) -> None:
        counter = _FakeFlopCounter(error=ValueError("no counts"))
        assert format_breakdown(counter) == {}
