"""DisKT (Dual-side Knowledge Tracing) model.

Separates each interaction into a familiarity side (correct) and an
unfamiliarity side (wrong) via contradictory attention, and contrasts the
two with a pairwise-distance regulariser.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """Scaled dot-product attention with an optional first-row zero-out."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    scores = scores.masked_fill(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    # Drop the query at position 0 so the first step sees no historical interaction.
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embedding_size)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    """Single transformer block: causal MHA + residual + FFN."""

    def __init__(self, embedding_size, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            embedding_size, d_feature, n_heads, dropout, kq_same=kq_same
        )

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, embedding_size)

        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout2 = nn.Dropout(dropout)

        # Causal masks owned by the layer, rebuilt on first forward / device move.
        self.register_buffer("_causal_mask_k0", None, persistent=False)
        self.register_buffer("_causal_mask_k1", None, persistent=False)
        self._const_seqlen = 0

    def _build_constants(self, seqlen: int, device: torch.device) -> None:
        ones = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device)
        self._causal_mask_k1 = ones.tril().view(1, 1, seqlen, seqlen)
        self._causal_mask_k0 = ones.tril(diagonal=-1).view(1, 1, seqlen, seqlen)
        self._const_seqlen = seqlen

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen = query.size(1)
        if (
            self._causal_mask_k0 is None
            or seqlen != self._const_seqlen
            or self._causal_mask_k0.device != query.device
        ):
            self._build_constants(seqlen, query.device)
        # mask=0: forbid the current step (knowledge retriever); mask=1: include it.
        src_mask = self._causal_mask_k0 if mask == 0 else self._causal_mask_k1
        zero_pad = mask == 0

        query2 = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=zero_pad
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """Stack of transformer blocks over question / interaction embeddings."""

    def __init__(
        self, num_blocks, embedding_size, d_ff, n_heads, dropout, kq_same, seq_len
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    embedding_size=embedding_size,
                    d_feature=embedding_size // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(num_blocks)
            ]
        )
        self.position_emb = CosinePositionalEmbedding(
            embedding_size=self.embedding_size, max_len=seq_len
        )

    def forward(self, q_embed_data, qa_embed_data):
        q_embed_data = q_embed_data + self.position_emb(q_embed_data)
        qa_embed_data = qa_embed_data + self.position_emb(qa_embed_data)

        x = q_embed_data
        y = qa_embed_data
        # mask=0 every layer: the current response never leaks into its own query.
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


def contradictory_attention(
    query, key, value1, value2, mask, dropout, counter_attention_mask
):
    """Attention that suppresses positions flagged as counterfactual."""
    bs, head, seqlen, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)
    device = query.device

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e32)
    p_attn = F.softmax(scores, dim=-1)

    # Zero out attention weights into counterfactual positions, then renormalise.
    p_attn = p_attn.masked_fill(counter_attention_mask[:, None, None, :] == 1, 0.0)
    if mask is not None:
        p_attn = p_attn.masked_fill(mask == 0, -1e32)
    p_attn = F.softmax(p_attn, dim=-1)

    pad_zero = torch.zeros(bs, head, 1, seqlen, device=device)
    p_attn = torch.cat([pad_zero, p_attn[:, :, 1:, :]], dim=2)
    if dropout is not None:
        p_attn = dropout(p_attn)

    output_v1 = torch.matmul(p_attn, value1)
    output_v2 = torch.matmul(p_attn, value2)
    return output_v1, output_v2, p_attn


class DualAttention(nn.Module):
    """Two-value attention that produces familiar / unfamiliar streams."""

    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_feature = d_model // n_heads
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v1, v2, counter_attention_mask):
        batch_size = q.size(0)
        seqlen = q.size(1)
        device = q.device
        src_mask = (
            torch.ones(seqlen, seqlen, dtype=torch.bool, device=device)
            .tril(diagonal=-1)
            .view(1, 1, seqlen, seqlen)
        )

        q = q.view(batch_size, -1, self.n_heads, self.n_feature).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.n_feature).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.n_heads, self.n_feature).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.n_heads, self.n_feature).transpose(1, 2)

        output_v1, output_v2, _ = contradictory_attention(
            q, k, v1, v2, src_mask, self.dropout, counter_attention_mask
        )
        output_v1 = (
            output_v1.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output_v2 = (
            output_v2.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        return output_v1, output_v2


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.dropout(self.activation(self.w_1(input_tensor)))
        hidden_states = self.dropout(self.w_2(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float()
            * -(math.log(10000.0) / embedding_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(1), :]


class DisKT(nn.Module):
    """DisKT model.

    Args:
        num_skills: Number of skill rows (concepts incl. padding row).
        num_questions: Number of questions for the Rasch difficulty embedding.
        seq_len: Maximum sequence length.
        embedding_size: Hidden dimension.
        num_blocks: Number of transformer blocks.
        dropout: Dropout probability.
        kq_same: If 1, key and query share the linear projection.
        d_ff: Feed-forward inner dimension inside each block.
        final_fc_dim: Width of the first output MLP layer.
        final_fc_dim2: Width of the second output MLP layer.
        num_attn_heads: Number of attention heads.
        separate_qr: Whether to embed question-response pairs separately.
        l2: Weight on the Rasch difficulty regulariser.
    """

    def __init__(
        self,
        num_skills,
        num_questions,
        seq_len,
        embedding_size,
        num_blocks,
        dropout,
        kq_same,
        d_ff=256,
        final_fc_dim=512,
        final_fc_dim2=256,
        num_attn_heads=8,
        separate_qr=False,
        l2=1e-5,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_questions = num_questions
        self.l2 = l2
        self.separate_qr = separate_qr
        embed_l = embedding_size

        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_skills + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)

        self.q_embed = nn.Embedding(self.num_skills, embed_l)
        if self.separate_qr:
            self.qa_embed = nn.Embedding(2 * self.num_skills + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(3, embed_l)

        self.model = Architecture(
            num_blocks=num_blocks,
            embedding_size=embedding_size,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=self.kq_same,
            seq_len=seq_len,
        )

        self.ffn = FeedForward(d_model=embed_l, inner_size=embed_l * 2, dropout=dropout)
        self.dual_attention = DualAttention(
            n_heads=num_attn_heads, d_model=embedding_size
        )
        self.out = nn.Sequential(
            nn.Linear(embedding_size + embed_l * 2, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        self.reset()

    def reset(self):
        """Initialise Rasch difficulty rows to zero."""
        if self.num_questions > 0:
            for p in self.parameters():
                if p.size(0) == self.num_questions + 1:
                    torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qr:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def rasch_emb(self, q_data, pid_data, target):
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.num_questions > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)
            qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
        else:
            pid_embed_data = q_embed_data
        return q_embed_data, qa_embed_data, pid_embed_data

    def forward(self, questions, skills, responses, mask, counter_attention_mask):
        """Run a forward pass.

        Args:
            questions: Question id sequence [B, S].
            skills: Collapsed concept id sequence [B, S] (0 = padding).
            responses: Response sequence [B, S] (0/1).
            mask: Valid-position mask [B, S].
            counter_attention_mask: Counterfactual flags [B, S].

        Returns:
            preds: Per-step correctness probability [B, S].
            reg_loss: Familiar / unfamiliar contrastive regulariser.
        """
        valid = mask.long()
        masked_r = responses * valid

        pos_q, pos_qa, _ = self.rasch_emb(
            masked_r * skills, masked_r * questions, 2 - masked_r
        )
        neg_q, neg_qa, _ = self.rasch_emb(
            (1 - masked_r) * skills, (1 - masked_r) * questions, 2 * masked_r
        )
        q_embed_data, qa_embed_data, pid_embed_data = self.rasch_emb(
            skills, questions, masked_r
        )

        y1, y2, y = pos_qa, neg_qa, qa_embed_data
        x = q_embed_data

        distance = F.pairwise_distance(
            y1.reshape(y1.size(0), -1), y2.reshape(y2.size(0), -1)
        )
        reg_loss = torch.mean(distance) * 0.001

        x = self.model(x, y)

        y1, y2 = self.ffn(y1), self.ffn(y2)
        y1, y2 = self.dual_attention(x, x, y1, y2, counter_attention_mask)

        x = x - (y1 + y2)
        x = x - pid_embed_data
        x = torch.cat([x, q_embed_data], dim=-1)
        x = torch.cat([x, y1 - y2], dim=-1)
        output = self.out(x).squeeze(-1)
        preds = torch.sigmoid(output)
        return preds, reg_loss
