"""Multihead attention without unused attention weight averaging."""

import torch
from torch import nn


def multihead_attention(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve gradients and dropout; use weight-free attention at inference."""
    output, _ = module(
        query,
        key,
        value,
        attn_mask=attn_mask,
        need_weights=module.training or torch.is_grad_enabled(),
        average_attn_weights=False,
    )
    return output
