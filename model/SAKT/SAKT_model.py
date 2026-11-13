"""SAKT (Self-Attentive Knowledge Tracing) model."""

import copy

import torch
from torch import nn


class TransformerFFN(nn.Module):
    """Feed-forward block for SAKT."""

    def __init__(self, emb_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAKTBlock(nn.Module):
    """SAKT attention block with causal self-attention."""

    def __init__(self, emb_size: int, num_attn_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(emb_size)

        self.ffn = TransformerFFN(emb_size, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(emb_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query_t = query.permute(1, 0, 2)
        key_t = key.permute(1, 0, 2)
        value_t = value.permute(1, 0, 2)

        attn_emb = self.attn(
            query_t,
            key_t,
            value_t,
            attn_mask=self._causal_mask(key_t.shape[0], key_t.device),
            need_weights=self.attn.training or torch.is_grad_enabled(),
            average_attn_weights=False,
        )[0]
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = attn_emb.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(query + attn_emb)

        ffn_emb = self.ffn(attn_emb)
        ffn_emb = self.ffn_dropout(ffn_emb)
        return self.ffn_layer_norm(attn_emb + ffn_emb)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


class SAKT(nn.Module):
    """Self-Attentive Knowledge Tracing.

    Forward semantics:
    - history concepts/responses are sequence[:, :-1] and response[:, :-1]
    - current concepts are sequence[:, 1:]
    - output[:, j] predicts response[:, j + 1]
    """

    def __init__(
        self,
        num_c: int,
        seq_len: int,
        emb_size: int,
        num_attn_heads: int,
        dropout: float,
        num_en: int = 2,
    ) -> None:
        super().__init__()
        if num_attn_heads <= 0:
            raise ValueError("num_attn_heads must be positive for SAKT")
        if num_en <= 0:
            raise ValueError("num_en must be positive for SAKT")
        if emb_size % num_attn_heads != 0:
            raise ValueError("emb_size must be divisible by num_attn_heads for SAKT")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 for shifted SAKT input")

        self.model_name = "sakt"
        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        self.interaction_emb = nn.Embedding(num_c * 2, emb_size)
        self.exercise_emb = nn.Embedding(num_c, emb_size)
        self.position_emb = nn.Embedding(seq_len, emb_size)

        self.blocks = _get_clones(SAKTBlock(emb_size, num_attn_heads, dropout), num_en)
        self.dropout_layer = nn.Dropout(dropout)
        self.pred = nn.Linear(emb_size, 1)

    def base_emb(
        self,
        history_concepts: torch.Tensor,
        history_responses: torch.Tensor,
        current_concepts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interaction_ids = history_concepts + self.num_c * history_responses
        current_emb = self.exercise_emb(current_concepts)
        interaction_emb = self.interaction_emb(interaction_ids)

        shift_len = interaction_emb.shape[1]
        if shift_len > self.seq_len:
            raise ValueError(
                f"SAKT received sequence length {shift_len}, but seq_len={self.seq_len}"
            )

        position_ids = torch.arange(shift_len, device=interaction_emb.device)
        position_emb = self.position_emb(position_ids).unsqueeze(0)
        interaction_emb = interaction_emb + position_emb
        return current_emb, interaction_emb

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.shape != response.shape:
            raise ValueError("sequence and response must have the same shape")

        if sequence.shape[1] < 2:
            return torch.empty(
                sequence.shape[0], 0, dtype=torch.float, device=sequence.device
            )

        history_concepts = sequence[:, :-1]
        history_responses = response[:, :-1]
        current_concepts = sequence[:, 1:]

        current_emb, interaction_emb = self.base_emb(
            history_concepts, history_responses, current_concepts
        )
        hidden = interaction_emb
        for block in self.blocks:
            hidden = block(current_emb, hidden, hidden)

        return torch.sigmoid(self.pred(self.dropout_layer(hidden))).squeeze(-1)
