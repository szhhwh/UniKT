"""Inject DJIM learning features into unchanged DKT/SAINT instances."""

from collections.abc import Callable

import torch
from torch import nn

from model.DKT.DKT_model import DKT
from model.SAINT.SAINT_model import SAINT


def _run_with_embedding_output(
    embedding: nn.Module,
    replacement: Callable[[torch.Tensor], torch.Tensor],
    run_model: Callable[[], torch.Tensor],
) -> torch.Tensor:
    calls = 0

    def inject(_module: nn.Module, _inputs: tuple, output: torch.Tensor):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise RuntimeError("DJIM expected exactly one embedding call")
        injected = replacement(output)
        if injected.shape != output.shape:
            raise ValueError("DJIM interaction features have the wrong shape")
        if injected.device != output.device or injected.dtype != output.dtype:
            raise ValueError("DJIM interaction features must match embedding type")
        return injected

    handle = embedding.register_forward_hook(inject)
    try:
        result = run_model()
    finally:
        handle.remove()
    if calls != 1:
        raise RuntimeError("DJIM embedding hook was not called")
    return result


def knowledge_prediction(
    model: DKT | SAINT,
    question: torch.Tensor,
    skill: torch.Tensor,
    response: torch.Tensor,
    valid: torch.Tensor,
    interaction: torch.Tensor,
) -> torch.Tensor:
    """Use the original model forward while substituting its input embedding."""
    if isinstance(model, DKT):
        return _run_with_embedding_output(
            model.interaction_emb,
            lambda _output: interaction,
            lambda: model(skill, response, valid),
        )
    if isinstance(model, SAINT):
        return _run_with_embedding_output(
            model.decoder[0].response_emb,
            lambda output: torch.cat((output[:, :1], interaction[:, :-1]), dim=1),
            lambda: model(question, skill, response),
        )
    raise TypeError(f"Unsupported DJIM knowledge model: {type(model).__name__}")
