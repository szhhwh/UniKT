"""Tests for the custom ``aten::_cudnn_rnn`` FLOP formula registration.

The formula is registered into ``torch.utils.flop_counter.flop_registry`` at
import of :mod:`utils.efficiency.measures`; here it is invoked directly with
hand-built shapes to verify the per-mode gate multiplication factors and the
layer/direction bookkeeping (1 FMA = 2 FLOPs).
"""

from collections.abc import Callable

import pytest
import torch
from torch.utils.flop_counter import flop_registry

from utils.efficiency.measures import custom_formulas

_CUDNN_RNN = torch.ops.aten._cudnn_rnn


def _formula() -> Callable[..., int]:
    assert _CUDNN_RNN in flop_registry, "cudnn RNN formula not registered"
    return flop_registry[_CUDNN_RNN]


def _run(
    *,
    mode: int,
    batch: int = 2,
    seq: int = 3,
    input_dim: int = 8,
    hidden: int = 16,
    num_layers: int = 1,
    bidirectional: bool = False,
    batch_first: bool = True,
) -> int:
    input_size = (batch, seq, input_dim) if batch_first else (seq, batch, input_dim)
    return _formula()(
        input_size=input_size,
        weight_sizes=None,
        weight_stride0=0,
        weight_buf_size=0,
        hx_size=0,
        cx_size=0,
        mode=mode,
        hidden_size=hidden,
        proj_size=0,
        num_layers=num_layers,
        batch_first=batch_first,
        dropout=0.0,
        train=True,
        bidirectional=bidirectional,
        batch_sizes=None,
        dropout_state_size=0,
    )


def _expected(
    gate_mult: int,
    batch: int,
    seq: int,
    input_dim: int,
    hidden: int,
    layers: int,
    directions: int,
) -> int:
    """input->hidden plus hidden->hidden matmuls, per gate, times 2 for FMA."""
    total = 0
    for layer in range(layers):
        layer_dim = input_dim if layer == 0 else hidden * directions
        for _ in range(directions):
            total += 2 * batch * seq * gate_mult * layer_dim * hidden
            total += 2 * batch * seq * gate_mult * hidden * hidden
    return total


class TestCudnnRnnFormula:
    def test_formula_registered_and_is_module_function(self) -> None:
        assert flop_registry[_CUDNN_RNN] is custom_formulas._cudnn_rnn_flop

    def test_lstm_mode_2_uses_four_gates(self) -> None:
        got = _run(mode=2)
        assert got == _expected(4, 2, 3, 8, 16, layers=1, directions=1)

    def test_gru_mode_3_uses_three_gates(self) -> None:
        got = _run(mode=3)
        assert got == _expected(3, 2, 3, 8, 16, layers=1, directions=1)

    @pytest.mark.parametrize("mode,gates", [(0, 1), (1, 1)])
    def test_rnn_modes_use_one_gate(self, mode: int, gates: int) -> None:
        assert _run(mode=mode) == _expected(gates, 2, 3, 8, 16, layers=1, directions=1)

    def test_bidirectional_doubles_and_feeds_layer_input(self) -> None:
        got = _run(mode=2, bidirectional=True, num_layers=2)
        assert got == _expected(4, 2, 3, 8, 16, layers=2, directions=2)

    def test_seq_first_layout_agrees_with_batch_first(self) -> None:
        assert _run(mode=2, batch_first=True) == _run(mode=2, batch_first=False)

    def test_unknown_mode_raises(self) -> None:
        # An unknown cuDNN mode code is a new cell variant the table does not
        # know; refuse to guess rather than silently mis-counting FLOPs.
        with pytest.raises(ValueError, match="Unknown cuDNN RNN mode code"):
            _run(mode=99)
