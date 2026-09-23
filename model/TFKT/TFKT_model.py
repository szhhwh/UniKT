import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from .behavior import KTBehaviorExtractor
from .causal_conv import CausalConv1d
from .kan import ChebyKANLinear


@lru_cache(maxsize=8)
def _causal_masks(length: int, device: torch.device):
    positions = torch.arange(length, device=device)
    current_and_past = positions[:, None] >= positions[None, :]
    past_only = positions[:, None] > positions[None, :]
    return current_and_past[None, None], past_only[None, None]


@lru_cache(maxsize=8)
def _position_effect(length: int, device: torch.device, dtype: torch.dtype):
    positions = torch.arange(length, device=device)
    return (positions[:, None] - positions[None, :]).abs()[None, None].to(dtype)


class TFKT(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        kq_same,
        dropout,
        final_fc_dim=512,
        n_heads=8,
        d_ff=2048,
        l2=1e-5,
        separate_qa=False,
    ):
        super().__init__()
        if n_question < 1 or n_pid < 0 or n_blocks < 1 or n_heads < 1:
            raise ValueError(
                "n_question, n_blocks and n_heads must be positive; n_pid must be nonnegative"
            )
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_question = n_question
        self.dropout = dropout
        self.n_pid = n_pid
        self.l2 = l2
        self.separate_qa = separate_qa
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, d_model)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, d_model)
        self.q_embed = nn.Embedding(self.n_question + 1, d_model)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, d_model)
        else:
            self.qa_embed = nn.Embedding(2, d_model)
        self.model = Architecture(
            n_blocks=n_blocks,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            kq_same=kq_same,
        )

        self.out = nn.Sequential(
            nn.Linear(2 * d_model, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )
        self.reset()

    def reset(self):
        if self.n_pid > 0:
            nn.init.zeros_(self.difficult_param.weight)

    def forward(self, sequence, response, mask=None, pid_data=None):
        q_data = sequence + 1
        if mask is not None:
            q_data = q_data.masked_fill(~mask.bool(), 0)
        if self.n_pid > 0 and pid_data is None:
            raise ValueError("pid_data is required when n_pid > 0")
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * response
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = response
            qa_embed_data = self.qa_embed(qa_data) + q_embed_data

        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )
            c_reg_loss = (pid_embed_data**2.0).sum() * self.l2
        else:
            c_reg_loss = 0.0

        d_output = self.model(q_embed_data, qa_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)

        return torch.sigmoid(output.squeeze(-1)), c_reg_loss


class Architecture(nn.Module):
    def __init__(
        self,
        n_blocks,
        d_model,
        d_ff,
        n_heads,
        dropout,
        kq_same,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.linear = nn.Linear(2 * d_model, d_model)
        self.alpha = nn.Linear(d_model, d_model)
        self.transG = nn.Linear(d_model, d_model)
        self.transS = nn.Linear(d_model, d_model)
        self.linear_com = nn.Linear(2 * d_model, d_model)
        self.norm_k = nn.LayerNorm(normalized_shape=d_model)
        self.norm_e = nn.LayerNorm(normalized_shape=d_model)
        self.norm_r = nn.LayerNorm(normalized_shape=2 * d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.causal = CausalConv1d(d_model, d_model, kernel_size=3)
        self.linear_feat = nn.Linear(3 * d_model, d_model)
        self.behavior_extractor = KTBehaviorExtractor()
        self.kan_linear = ChebyKANLinear(d_model, d_model, 3)
        self.gate_param = nn.Parameter(torch.zeros(1))

        self.blocks_1 = nn.ModuleList(
            TransformerLayer(
                d_model, d_model // n_heads, d_ff, n_heads, dropout, kq_same
            )
            for _ in range(n_blocks)
        )
        self.blocks_2 = nn.ModuleList(
            TransformerLayer(
                d_model, d_model // n_heads, d_ff, n_heads, dropout, kq_same
            )
            for _ in range(n_blocks * 2)
        )

    def forward(self, q_embed_data, qa_embed_data):
        y = qa_embed_data
        x = q_embed_data
        q_alpha = self.sigmoid(self.alpha(q_embed_data))

        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)
            guess = torch.sigmoid(self.transG(y))
            slip = torch.sigmoid(self.transS(y))
            y = y.transpose(1, 2)
            y3 = self.causal(y).transpose(1, 2)
            y1 = self.conv1(y).transpose(1, 2)
            y = torch.cat((y1, y3), dim=-1)
            y = self.norm_r(y)
            y = self.linear(y)
            y = y * torch.sigmoid(y)
            y = (1 - y) * guess + y * (1 - slip)
            y = self.dropout(y)

        for idx in range(0, len(self.blocks_2), 2):
            question_block = self.blocks_2[idx]
            response_block = self.blocks_2[idx + 1]

            x = question_block(mask=1, query=x, key=x, values=x, apply_pos=False)
            x_residual = x
            x = self.norm_e(x)
            x = self.kan_linear(x)
            x = self.dropout(x)
            gate = torch.sigmoid(self.gate_param)
            x = x_residual + gate * x
            x = self.norm_k(x)

            x = response_block(mask=0, query=x, key=x, values=y, apply_pos=True)
            x = response_block(mask=0, query=x, key=y, values=y, apply_pos=True)
            x_residual = x
            feat_low, feat_mid, feat_high = self.behavior_extractor(x)
            x = torch.cat((feat_low, feat_mid, feat_high), dim=-1)
            x = self.gelu(self.linear_feat(x))
            x = self.gelu(self.linear_com(torch.cat((x_residual, x), dim=-1)))
            x = response_block(mask=0, query=x, key=x, values=x, apply_pos=True)

        return torch.sigmoid(1.7 * q_alpha * x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same == 1
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """mask=1 includes the current step; mask=0 allows only earlier steps."""
        current_and_past, past_only = _causal_masks(query.size(1), query.device)
        src_mask = past_only if mask == 0 else current_and_past
        query2 = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=mask == 0
        )
        query = self.layer_norm1(query + self.dropout1(query2))
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = self.layer_norm2(query + self.dropout2(query2))
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if not self.kq_same:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        batch_size = q.size(0)
        k_input = k
        k = self.k_linear(k_input).view(batch_size, -1, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        elif q is k_input:
            q = k
        else:
            q = self.k_linear(q).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)
        scores = attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            self.gammas,
        )
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(concat)


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    batch_size, heads, length = scores.shape[:3]

    with torch.no_grad():
        masked_scores = scores.masked_fill(~mask, -1e32)
        masked_scores = F.softmax(masked_scores, dim=-1) * mask.float()
        cumulative = torch.cumsum(masked_scores, dim=-1)
        total = masked_scores.sum(dim=-1, keepdim=True)
        distance = _position_effect(length, scores.device, scores.dtype)
        distance = ((total - cumulative) * distance).clamp(min=0.0).sqrt()

    decay = -F.softplus(gamma).unsqueeze(0)
    effect = (distance * decay).exp().clamp(min=1e-5, max=1e5)
    scores = (scores * effect).masked_fill(~mask, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        first_row = scores.new_zeros(batch_size, heads, 1, length)
        scores = torch.cat((first_row, scores[:, :, 1:, :]), dim=2)
    return torch.matmul(dropout(scores), v)
