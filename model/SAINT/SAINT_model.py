"""SAINT (Separated Self-Attentive Knowledge Tracing) model.

Choi et al., "Integrating Temporal Features for EdNet Correctness
Prediction", AAAI 2021 (arXiv:2002.07175).
"""

import copy

import torch
from torch import nn

from utils.attention import multihead_attention


class TransformerFFN(nn.Module):
    """Linear-ReLU-Dropout-Linear feed-forward network."""

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


def _ut_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Causal attention mask: True blocks key j > query i."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    # deepcopy: blocks share one random init, then train independently
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


class SAINTEncoderBlock(nn.Module):
    """Causal self-attention block over (question, skill, position) embeddings."""

    def __init__(
        self,
        emb_size: int,
        num_attn_heads: int,
        num_questions: int,
        num_skills: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.question_emb = nn.Embedding(num_questions, emb_size)
        self.skill_emb = nn.Embedding(num_skills, emb_size)

        self.self_attn = nn.MultiheadAttention(
            emb_size, num_attn_heads, dropout=dropout
        )
        self.attn_layer_norm = nn.LayerNorm(emb_size)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = TransformerFFN(emb_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(emb_size)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        question: torch.Tensor,
        skill: torch.Tensor,
        position_emb: torch.Tensor,
        first_block: bool,
    ) -> torch.Tensor:
        if first_block:
            out = self.question_emb(question) + self.skill_emb(skill) + position_emb
        else:
            out = question

        out = out.permute(1, 0, 2)  # (b, n, d) -> (n, b, d)
        n = out.shape[0]

        # norm-first residual
        out = self.attn_layer_norm(out)
        skip_out = out
        out = multihead_attention(
            self.self_attn, out, out, out, attn_mask=_ut_mask(n, out.device)
        )
        out = self.attn_dropout(out)
        out = out + skip_out

        out = out.permute(1, 0, 2)  # (n, b, d) -> (b, n, d)
        out = self.ffn_layer_norm(out)
        skip_out = out
        out = self.ffn(out)
        out = self.ffn_dropout(out)
        out = out + skip_out

        return out


class SAINTDecoderBlock(nn.Module):
    """Decoder block: masked self-attention, cross-attention, FFN."""

    def __init__(self, emb_size: int, num_attn_heads: int, dropout: float) -> None:
        super().__init__()
        # 3 rows: response ids {0, 1} plus the start token id 2
        self.response_emb = nn.Embedding(3, emb_size)

        self.self_attn = nn.MultiheadAttention(
            emb_size, num_attn_heads, dropout=dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(emb_size)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            emb_size, num_attn_heads, dropout=dropout
        )
        self.cross_attn_layer_norm = nn.LayerNorm(emb_size)
        self.cross_attn_dropout = nn.Dropout(dropout)

        self.ffn = TransformerFFN(emb_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(emb_size)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        response: torch.Tensor,
        position_emb: torch.Tensor,
        encoder_out: torch.Tensor,
        first_block: bool,
    ) -> torch.Tensor:
        out = self.response_emb(response) + position_emb if first_block else response

        out = out.permute(1, 0, 2)  # (b, n, d) -> (n, b, d)
        n = out.shape[0]

        out = self.self_attn_layer_norm(out)
        skip_out = out
        out = multihead_attention(
            self.self_attn, out, out, out, attn_mask=_ut_mask(n, out.device)
        )
        out = self.self_attn_dropout(out)
        out = skip_out + out

        # each block re-normalises the encoder output
        encoder_out = encoder_out.permute(1, 0, 2)  # (b, n, d) -> (n, b, d)
        encoder_out = self.cross_attn_layer_norm(encoder_out)
        skip_out = out
        out = multihead_attention(
            self.cross_attn,
            out,
            encoder_out,
            encoder_out,
            attn_mask=_ut_mask(n, out.device),
        )
        out = self.cross_attn_dropout(out)
        out = out + skip_out

        out = out.permute(1, 0, 2)  # (n, b, d) -> (b, n, d)
        out = self.ffn_layer_norm(out)
        skip_out = out
        out = self.ffn(out)
        out = self.ffn_dropout(out)
        out = out + skip_out

        return out


class SAINT(nn.Module):
    """SAINT encoder-decoder transformer.

    Forward semantics:
    - decoder input at position t is the start token (t=0) or response t-1
    - ``out[:, t]`` predicts ``response[:, t]``
    """

    START_TOKEN_ID = 2

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        seq_len: int,
        emb_size: int,
        num_attn_heads: int,
        dropout: float,
        n_blocks: int = 1,
    ) -> None:
        super().__init__()
        if num_attn_heads <= 0:
            raise ValueError("num_attn_heads must be positive for SAINT")
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive for SAINT")
        if emb_size % num_attn_heads != 0:
            raise ValueError("emb_size must be divisible by num_attn_heads for SAINT")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 for shifted SAINT decoder")

        self.model_name = "saint"
        self.num_q = num_questions
        self.num_c = num_skills
        self.n_blocks = n_blocks

        self.position_emb = nn.Embedding(seq_len, emb_size)
        self.encoder = _get_clones(
            SAINTEncoderBlock(
                emb_size, num_attn_heads, num_questions, num_skills, dropout
            ),
            n_blocks,
        )
        self.decoder = _get_clones(
            SAINTDecoderBlock(emb_size, num_attn_heads, dropout), n_blocks
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_size, 1)

    def forward(
        self,
        question: torch.Tensor,
        skill: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        if question.shape != skill.shape or question.shape != response.shape:
            raise ValueError(
                "question, skill and response must have the same shape for SAINT"
            )

        seq_len = question.shape[1]
        position_ids = torch.arange(seq_len, device=question.device)
        position_emb = self.position_emb(position_ids).unsqueeze(0)  # (1, n, d)

        # first block embeds raw ids; later blocks pass hidden states through
        hidden = question
        for i, block in enumerate(self.encoder):
            hidden = block(
                hidden,
                skill if i == 0 else hidden,
                position_emb,
                first_block=(i == 0),
            )
        encoder_out = hidden

        # right shift: [start, r_0, ..., r_{S-2}]
        start_token = torch.full(
            (response.shape[0], 1),
            self.START_TOKEN_ID,
            dtype=response.dtype,
            device=response.device,
        )
        decoder_in = torch.cat((start_token, response[:, :-1]), dim=1)

        for i, block in enumerate(self.decoder):
            decoder_in = block(
                decoder_in, position_emb, encoder_out, first_block=(i == 0)
            )

        out = self.out(self.dropout(decoder_in))
        return torch.sigmoid(out).squeeze(-1)
