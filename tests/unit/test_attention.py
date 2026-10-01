import copy
from typing import Any

import pytest
import torch
from torch import nn

from utils.attention import multihead_attention


@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_mha_preserves_outputs_gradients_and_rng(
    dropout: float, input_grad: bool, training: bool
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        original = nn.MultiheadAttention(16, 4, dropout=dropout).train(training)
        optimized = copy.deepcopy(original)
        data = torch.randn(7, 2, 16)
        mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
        results = []
        for module, use_helper in ((original, False), (optimized, True)):
            inputs = data.clone().requires_grad_(input_grad)
            torch.manual_seed(71)
            if use_helper:
                output = multihead_attention(module, inputs, inputs, inputs, mask)
            else:
                output, _ = module(inputs, inputs, inputs, attn_mask=mask)
            output.square().mean().backward()
            gradients: list[torch.Tensor] = []
            for parameter in module.parameters():
                gradient = parameter.grad
                assert gradient is not None
                gradients.append(gradient.clone())
            results.append(
                (
                    output.detach(),
                    gradients,
                    inputs.grad,
                    torch.get_rng_state(),
                )
            )
        before, after = results
        rtol, atol = (0, 0) if training else (1e-6, 2e-7)
        torch.testing.assert_close(before[0], after[0], rtol=rtol, atol=atol)
        for a, b in zip(before[1], after[1], strict=True):
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        if input_grad:
            torch.testing.assert_close(before[2], after[2], rtol=rtol, atol=atol)
        assert torch.equal(before[3], after[3])


@pytest.mark.parametrize("training", [False, True])
def test_mha_does_not_recompute_in_backward(training: bool) -> None:
    with torch.random.fork_rng(devices=[]):
        module = nn.MultiheadAttention(16, 4).train(training)
        inputs = torch.randn(7, 2, 16, requires_grad=True)
        mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
        calls: list[None] = []
        handle = module.register_forward_pre_hook(lambda *args: calls.append(None))
        try:
            output = multihead_attention(module, inputs, inputs, inputs, mask)
            assert len(calls) == 1
            output.square().mean().backward()
            assert len(calls) == 1
        finally:
            handle.remove()


def test_mha_inference_uses_weight_free_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        module = nn.MultiheadAttention(16, 4).eval()
        inputs = torch.randn(7, 2, 16)
        mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
        forward = module.forward
        flags: list[bool] = []

        def record_forward(
            *args: Any, **kwargs: Any
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            flags.append(kwargs["need_weights"])
            return forward(*args, **kwargs)

        with torch.inference_mode():
            expected, _ = module(inputs, inputs, inputs, attn_mask=mask)
            monkeypatch.setattr(module, "forward", record_forward)
            actual = multihead_attention(module, inputs, inputs, inputs, mask)
        assert flags == [False]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
