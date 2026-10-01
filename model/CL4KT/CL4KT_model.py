"""CL4KT (Contrastive Learning for Knowledge Tracing) model."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class Similarity(nn.Module):
    """Cosine similarity scaled by a temperature, serving as SimCSE logits."""

    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(x, y, dim=-1) / self.temp


def individual_attention(q, k, v, d_k, mask, dropout, gamma, position_effect):
    """Monotonic attention with a learnable distance decay.

    Args:
        q, k, v: [B, h, S, d_k].
        mask: [1, 1, S, S] bool attention mask.
        gamma: [h, 1, 1] learnable decay parameter.
        position_effect: [1, 1, S, S] precomputed |i - j| distance matrix.

    Returns:
        output [B, h, S, d_k], attn_scores [B, h, S, S].
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_.masked_fill_(mask == 0, 0.0)
        dist_scores = torch.cumsum(scores_, dim=-1)
        total = scores_.sum(dim=-1, keepdim=True)
        del scores_
        torch.sub(total, dist_scores, out=dist_scores)
        del total
        dist_scores.mul_(position_effect).clamp_min_(0.0).sqrt_()

    gamma_decay = (-F.softplus(gamma)).unsqueeze(0)  # [1, h, 1, 1]
    total_effect = torch.clamp((dist_scores * gamma_decay).exp(), min=1e-5, max=1e5)

    del dist_scores
    scores = scores * total_effect
    del total_effect
    scores = scores.masked_fill(mask == 0, -1e32)
    attn_scores = F.softmax(scores, dim=-1)
    output = torch.matmul(dropout(attn_scores), v)
    return output, attn_scores


class MultiHeadAttention(nn.Module):
    """Multi-head attention with the monotonic distance decay."""

    def __init__(self, d_model, n_heads, dropout, kq_same):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in (self.k_linear, self.v_linear):
            xavier_uniform_(lin.weight)
            constant_(lin.bias, 0.0)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)
            constant_(self.q_linear.bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, position_effect):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        if self.kq_same:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        else:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        output, attn = individual_attention(
            q, k, v, self.d_k, mask, self.dropout, self.gammas, position_effect
        )
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat), attn


class CL4KTTransformerLayer(nn.Module):
    """Transformer block with residual connections and a GELU feed-forward net."""

    def __init__(self, d_model, n_heads, d_ff, dropout, kq_same):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, kq_same)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, position_effect, apply_pos=True):
        query2, attn = self.attn(query, key, values, mask, position_effect)
        query = self.norm1(query + self.dropout1(query2))
        if apply_pos:
            query2 = self.linear2(self.ff_dropout(self.activation(self.linear1(query))))
            query = self.norm2(query + self.dropout2(query2))
        return query, attn


class CL4KT(nn.Module):
    """CL4KT model: causal question/interaction encoders with a knowledge retriever.

    The retriever attends the question stream over the interaction stream to
    produce next-response predictions. Training adds a SimCSE contrastive loss
    over two augmented views of each sequence.

    Args:
        num_skills: Number of skills (valid ids 0..num_skills-1).
        hidden_size: Hidden dimension.
        num_blocks: Transformer blocks per encoder.
        num_attn_heads: Attention heads.
        kq_same: Share the key/query projection when true.
        final_fc_dim: Output MLP width.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        temp: Contrastive similarity temperature.
        hard_negative_weight: Weight added to hard-negative logits.
        negative_prob: Augmentation response-flip probability (>0 enables hard negatives).
    """

    def __init__(
        self,
        num_skills: int,
        hidden_size: int = 64,
        num_blocks: int = 2,
        num_attn_heads: int = 8,
        kq_same: bool = True,
        final_fc_dim: int = 512,
        d_ff: int = 1024,
        dropout: float = 0.2,
        temp: float = 0.05,
        hard_negative_weight: float = 1.0,
        negative_prob: float = 1.0,
    ):
        super().__init__()
        self.num_skills = num_skills
        # BERT-style mask token sits one past the last valid skill id.
        self.mask_token_id = num_skills
        self.negative_prob = negative_prob
        self.hard_negative_weight = hard_negative_weight

        self.question_embed = nn.Embedding(num_skills + 1, hidden_size)
        self.interaction_embed = nn.Embedding(2 * (num_skills + 1), hidden_size)
        self.sim = Similarity(temp)

        self.question_encoder = nn.ModuleList(
            [
                CL4KTTransformerLayer(
                    hidden_size, num_attn_heads, d_ff, dropout, kq_same
                )
                for _ in range(num_blocks)
            ]
        )
        self.interaction_encoder = nn.ModuleList(
            [
                CL4KTTransformerLayer(
                    hidden_size, num_attn_heads, d_ff, dropout, kq_same
                )
                for _ in range(num_blocks)
            ]
        )
        self.knowledge_retriever = nn.ModuleList(
            [
                CL4KTTransformerLayer(
                    hidden_size, num_attn_heads, d_ff, dropout, kq_same
                )
                for _ in range(num_blocks)
            ]
        )

        self.out = nn.Sequential(
            nn.Linear(2 * hidden_size, final_fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, final_fc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim // 2, 1),
        )

        self.cl_loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # Seqlen-dependent attention constants. Non-persistent so they migrate
        # with .to(device) yet stay out of the checkpoint.
        self.register_buffer("_mask_diag0", None, persistent=False)
        self.register_buffer("_mask_diag1", None, persistent=False)
        self.register_buffer("_mask_full", None, persistent=False)
        self.register_buffer("_position_effect", None, persistent=False)
        self._const_seqlen = 0

    def _build_constants(self, seqlen: int, device: torch.device) -> None:
        ones = torch.ones(seqlen, seqlen, device=device)
        self._mask_diag1 = ones.tril().view(1, 1, seqlen, seqlen).bool()
        self._mask_diag0 = ones.tril(diagonal=-1).view(1, 1, seqlen, seqlen).bool()
        self._mask_full = ones.view(1, 1, seqlen, seqlen).bool()
        idx = torch.arange(seqlen, device=device, dtype=torch.float32)
        self._position_effect = torch.abs(
            idx.view(seqlen, 1) - idx.view(1, seqlen)
        ).view(1, 1, seqlen, seqlen)
        self._const_seqlen = seqlen

    def _get_constants(self, seqlen: int, device: torch.device):
        if (
            self._position_effect is None
            or seqlen != self._const_seqlen
            or self._position_effect.device != device
        ):
            self._build_constants(seqlen, device)
        return (
            self._mask_diag0,
            self._mask_diag1,
            self._mask_full,
            self._position_effect,
        )

    def _interaction_embed(self, skills: torch.Tensor, responses: torch.Tensor):
        interactions = skills + self.num_skills * responses
        return self.interaction_embed(interactions)

    def _mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        return (x * weights).sum(1) / weights.sum(1).clamp(min=1e-9)

    def predict(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Run the causal encoders and retriever; return per-position correctness.

        Args:
            q: Skill id sequence [B, S].
            r: Response sequence [B, S].

        Returns:
            Correctness probabilities [B, S]; position t predicts response t.
        """
        _, mask_diag1, _, pe = self._get_constants(q.size(1), q.device)
        mask_diag0 = self._mask_diag0

        q_embed = self.question_embed(q)
        i_embed = self._interaction_embed(q, r)

        x, y = q_embed, i_embed
        for block in self.question_encoder:
            x, _ = block(mask_diag1, x, x, x, pe)
        for block in self.interaction_encoder:
            y, _ = block(mask_diag1, y, y, y, pe)
        for block in self.knowledge_retriever:
            x, attn = block(mask_diag0, x, x, y, pe)

        retrieved = torch.cat([x, q_embed], dim=-1)
        output = torch.sigmoid(self.out(retrieved)).squeeze(-1)
        return output

    def compute_cl_loss(
        self,
        q: torch.Tensor,
        q_i: torch.Tensor,
        q_j: torch.Tensor,
        r_i: torch.Tensor,
        r_j: torch.Tensor,
        neg_r: torch.Tensor,
        mask_i: torch.Tensor,
        mask_j: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """SimCSE contrastive loss over two augmented views of the batch.

        Each sample's view_i is the positive of the same sample's view_j; the
        batch forms the in-batch negatives. Interaction contrast additionally
        uses response-flipped sequences as hard negatives.
        """
        _, _, mask_full, pe = self._get_constants(q_i.size(1), q_i.device)

        ques_i = self.question_embed(q_i)
        ques_j = self.question_embed(q_j)
        inter_i = self._interaction_embed(q_i, r_i)
        inter_j = self._interaction_embed(q_j, r_j)

        for block in self.question_encoder:
            ques_i, _ = block(mask_full, ques_i, ques_i, ques_i, pe, apply_pos=False)
            ques_j, _ = block(mask_full, ques_j, ques_j, ques_j, pe, apply_pos=False)
        for block in self.interaction_encoder:
            inter_i, _ = block(
                mask_full, inter_i, inter_i, inter_i, pe, apply_pos=False
            )
            inter_j, _ = block(
                mask_full, inter_j, inter_j, inter_j, pe, apply_pos=False
            )

        pooled_ques_i = self._mean_pool(ques_i, mask_i)
        pooled_ques_j = self._mean_pool(ques_j, mask_j)
        ques_cos = self.sim(pooled_ques_i.unsqueeze(1), pooled_ques_j.unsqueeze(0))
        ques_labels = torch.arange(ques_cos.size(0), device=q_i.device)
        question_cl_loss = self.cl_loss_fn(ques_cos, ques_labels)

        pooled_inter_i = self._mean_pool(inter_i, mask_i)
        pooled_inter_j = self._mean_pool(inter_j, mask_j)
        inter_cos = self.sim(pooled_inter_i.unsqueeze(1), pooled_inter_j.unsqueeze(0))

        if self.negative_prob > 0:
            inter_k = self._interaction_embed(q, neg_r)
            for block in self.interaction_encoder:
                inter_k, _ = block(
                    mask_full, inter_k, inter_k, inter_k, pe, apply_pos=False
                )
            pooled_inter_k = self._mean_pool(inter_k, mask)
            neg_cos = self.sim(pooled_inter_i.unsqueeze(1), pooled_inter_k.unsqueeze(0))
            inter_cos = torch.cat([inter_cos, neg_cos], dim=1)

            batch_size = inter_cos.size(0)
            weights = torch.zeros_like(inter_cos)
            diag = torch.arange(batch_size, device=q_i.device)
            weights[diag, batch_size + diag] = self.hard_negative_weight
            inter_cos = inter_cos + weights

        inter_labels = torch.arange(inter_cos.size(0), device=q_i.device)
        interaction_cl_loss = self.cl_loss_fn(inter_cos, inter_labels)
        return question_cl_loss + interaction_cl_loss
