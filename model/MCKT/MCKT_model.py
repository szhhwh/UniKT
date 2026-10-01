"""MCKT model.

This module contains the project's local port of the MCKT reference
implementation.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def attention_score(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])

    seq = scores.shape[-1]
    x1 = torch.arange(seq, device=query.device, dtype=torch.float32).unsqueeze(-1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask, -1e9)
        scores_ = F.softmax(scores_, dim=-1)
        dist_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        del scores_
        torch.sub(disttotal_scores, dist_scores, out=dist_scores)
        del disttotal_scores
        dist_scores.mul_(torch.abs(x1 - x2)[None, None, :, :]).clamp_min_(0.0).sqrt_()

    gamma = -1.0 * gamma.abs().unsqueeze(0)
    total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

    del dist_scores
    scores = scores * total_effect
    del total_effect
    scores = scores.masked_fill(mask, -1e9)
    scores = torch.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask, 0)

    output = torch.matmul(scores, value)
    return output, scores


class MultiHeadForgetAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float, n_heads: int):
        super().__init__()

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = query.shape[0]
        origin_d = query.shape[-1]
        d_k = origin_d // self.n_heads
        query = self.q_linear(query).view(batch, -1, self.n_heads, d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch, -1, self.n_heads, d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch, -1, self.n_heads, d_k).transpose(1, 2)
        out, attn = attention_score(query, key, value, mask, self.gammas)
        out = out.transpose(1, 2).contiguous().view(batch, -1, origin_d)
        out = self.linear_out(out)
        return out, attn


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float, n_heads: int):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.attn = MultiHeadForgetAttention(d_model, dropout, n_heads)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        apply: bool = True,
    ) -> torch.Tensor:
        out, _ = self.attn(q, k, v, mask)
        q = q + self.dropout(out)
        q = self.layer_norm1(q)
        if apply:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
            q = q + self.dropout(query2)
            q = self.layer_norm2(q)
        return q


class MCKT(nn.Module):
    """Multi-level Contrastive Learning for Knowledge Tracing."""

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        d_model: int = 128,
        dropout: float = 0.1,
        n_heads: int = 8,
        temperature: float = 0.8,
        sim_threshold: float = 0.8,
        cl_batch_size: int = 10000,
        cl_exp_mode: str = "source",
        pro_loss_weight: float = 1.0,
        react_loss_weight: float = 1.0,
        state_loss_weight: float = 0.0001,
        pos_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.num_questions = num_questions
        self.num_skills = num_skills
        self.temperature = temperature
        self.sim_threshold = sim_threshold
        self.cl_batch_size = cl_batch_size
        if cl_exp_mode not in {"paper", "source"}:
            raise ValueError("cl_exp_mode must be either 'paper' or 'source'")
        self.cl_exp_mode = cl_exp_mode
        self.pro_loss_weight = pro_loss_weight
        self.react_loss_weight = react_loss_weight
        self.state_loss_weight = state_loss_weight

        self.pro_embed = nn.Parameter(torch.rand(num_questions, d_model))
        # Retained for parameter parity with the released MCKT source.
        self.skill_embed = nn.Parameter(torch.rand(num_skills, d_model))
        self.diff_embed = nn.Parameter(torch.rand(num_questions, 1))
        self.pro_change = nn.Parameter(torch.rand(num_skills, d_model))
        self.ans_embed = nn.Parameter(torch.rand(2, d_model))

        self.encoder = TransformerLayer(d_model, dropout, n_heads)
        self.decoder_1 = TransformerLayer(d_model, dropout, n_heads)
        self.decoder_2 = TransformerLayer(d_model, dropout, n_heads)

        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, 1),
        )

        if pos_matrix is None:
            pos_matrix = torch.eye(num_questions, dtype=torch.float32)
        self.register_buffer("pos_matrix", pos_matrix.float(), persistent=False)

    def _source_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = torch.matmul(a, b.transpose(-1, -2))
        return torch.exp(sim / self.temperature)

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return torch.matmul(a, b.transpose(-1, -2))

    def batched_semi_loss(self, z1: torch.Tensor, batch_size: int) -> torch.Tensor:
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes, device=device)
        losses = []
        pos_matrix = self.pos_matrix.to(device=device, dtype=z1.dtype)

        for i in range(num_batches):
            node_idx = indices[i * batch_size : (i + 1) * batch_size]
            refl_sim = self._source_similarity(z1[node_idx], z1)
            if self.cl_exp_mode == "source":
                refl_sim = torch.exp(refl_sim / self.temperature)
            now_use_matrix = pos_matrix[node_idx]

            numerator = (refl_sim * now_use_matrix).sum(dim=-1, keepdim=True)
            denominator = refl_sim.sum(dim=-1, keepdim=True)
            losses.append(-torch.log(numerator / denominator))

        return torch.cat(losses).mean()

    def pro_similar(self, pro_embed: torch.Tensor) -> torch.Tensor:
        return self._cosine_similarity(pro_embed, pro_embed)

    def contrast_state_cl(
        self,
        h_state: torch.Tensor,
        other_state: torch.Tensor,
        next_mask: torch.Tensor,
    ) -> torch.Tensor:
        cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")

        batch_seq = next_mask.shape[-1]
        batch_min_seq = int(next_mask.sum(dim=-1).min().item())
        if batch_min_seq <= 0:
            return h_state.new_tensor(0.0)

        temp_h = h_state.transpose(0, 1)
        neg_sim = self._source_similarity(temp_h, other_state.transpose(0, 1))
        labels = torch.arange(h_state.size(0), device=h_state.device).long()

        losses = []
        for i in range(batch_seq - batch_min_seq, batch_seq):
            losses.append(cl_loss_fn(neg_sim[i], labels))

        if not losses:
            return h_state.new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(
        self,
        last_problem: torch.Tensor,
        last_ans: torch.Tensor,
        next_problem: torch.Tensor,
        next_ans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_mask = next_ans != -1
        last_ans = last_ans.masked_fill(last_ans < 0, 0).long()
        next_ans = next_ans.masked_fill(next_ans < 0, 0).long()

        device = last_problem.device
        seq = last_problem.shape[-1]
        pro_embed = self.pro_embed

        # Corpus-level contrastive terms only serve the training loss; skip
        # them at eval (they are batch-independent and costly).
        if self.training:
            pro_loss = self.pro_loss_weight * self.batched_semi_loss(
                pro_embed, self.cl_batch_size
            )
            react_loss = self.react_loss_weight * self.batched_semi_loss(
                pro_embed + self.ans_embed[1].unsqueeze(0), self.cl_batch_size
            )
        else:
            pro_loss = react_loss = None

        next_pro_embed = F.embedding(next_problem, pro_embed)
        last_pro_embed = F.embedding(last_problem, pro_embed)

        next_x = next_pro_embed + F.embedding(next_ans, self.ans_embed)
        last_x = last_pro_embed + F.embedding(last_ans, self.ans_embed)

        pro_sim = self.pro_similar(next_pro_embed)
        pro_mask = (pro_sim < self.sim_threshold).unsqueeze(1)

        future_mask = (
            torch.triu(torch.ones((seq, seq), device=device), 1)
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        future_or_current_mask = (
            torch.triu(torch.ones((seq, seq), device=device), 0)
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        mask = future_mask | pro_mask
        l_mask = future_or_current_mask | pro_mask

        encoder_out = self.encoder(next_pro_embed, next_pro_embed, next_pro_embed, mask)
        f2 = self.decoder_1(next_x, next_x, next_x, mask, False)
        decoder_out = self.decoder_2(encoder_out, f2, f2, l_mask)

        next_state, _ = self.lstm(self.dropout(last_x))
        now_use = torch.cat([decoder_out, next_state, next_pro_embed], dim=-1)
        predict = torch.sigmoid(self.out(self.dropout(now_use))).squeeze(-1)

        if self.training:
            state_loss = self.contrast_state_cl(
                next_state, decoder_out, next_mask
            ) + self.contrast_state_cl(decoder_out, next_state, next_mask)
            state_loss = state_loss * self.state_loss_weight
        else:
            state_loss = None

        return predict, state_loss, pro_loss, react_loss
