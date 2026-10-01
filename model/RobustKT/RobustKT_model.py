import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class CausalTemporalConv(nn.Module):
    """One-dimensional temporal convolution that never looks ahead."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        crop = self.conv.padding[0]
        if crop == 0:
            return out
        return out[:, :, :-crop]


class Smooth(nn.Module):
    """Causal smoothing plus residual fusion used by RobustKT."""

    def __init__(self, dropout: float, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalTemporalConv(hidden_size, hidden_size, kernel_size)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        conv_input = input_tensor.permute(0, 2, 1)
        trend = self.causal_conv(conv_input).permute(0, 2, 1)
        random = input_tensor - trend
        smoothed = trend + (self.sqrt_beta**2) * random
        hidden_states = self.out_dropout(smoothed)
        return self.layer_norm(hidden_states + input_tensor)


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normed + self.beta


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_k: int,
    mask: torch.Tensor,
    dropout: nn.Dropout,
    zero_pad: bool,
    gamma: torch.Tensor | None = None,
    pdiff: torch.Tensor | None = None,
    pos_effect: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    heads = scores.size(1)

    with torch.no_grad():
        masked_scores = scores.masked_fill(mask == 0, -1e32)
        masked_scores = F.softmax(masked_scores, dim=-1)
        masked_scores.masked_fill_(mask == 0, 0.0)
        dist_scores = torch.cumsum(masked_scores, dim=-1)
        disttotal_scores = torch.sum(masked_scores, dim=-1, keepdim=True)
        del masked_scores
        torch.sub(disttotal_scores, dist_scores, out=dist_scores)
        del disttotal_scores
        dist_scores.mul_(pos_effect).clamp_min_(0.0).sqrt_()

    if gamma is None:
        gamma = torch.zeros(heads, 1, 1, device=q.device)
    gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
    if pdiff is None:
        total_effect = torch.clamp(
            (dist_scores * gamma).exp(),
            min=1e-5,
            max=1e5,
        )
    else:
        diff = pdiff.unsqueeze(1).expand(
            pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2]
        )
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(
            (dist_scores * gamma * diff).exp(),
            min=1e-5,
            max=1e5,
        )
    del dist_scores
    scores = scores * total_effect
    del total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RobustKT decay distance effect."""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        bias: bool = True,
        emb_type: str = "qid",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
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

    def _reset_parameters(self) -> None:
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        zero_pad: bool,
        pdiff: torch.Tensor | None = None,
        pos_effect: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = q.size(0)

        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        if "pdiff" not in self.emb_type:
            pdiff = None
        scores = attention(
            q,
            k,
            v,
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            self.gammas,
            pdiff,
            pos_effect=pos_effect,
        )
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    """Transformer layer used by RobustKT's two-stream architecture."""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
        emb_type: str,
    ) -> None:
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=kq_same == 1,
            emb_type=emb_type,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        causal_mask: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        values: torch.Tensor,
        zero_pad: bool,
        apply_pos: bool = True,
        pos_effect: torch.Tensor | None = None,
        pdiff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query2 = self.masked_attn_head(
            query,
            key,
            values,
            mask=causal_mask,
            zero_pad=zero_pad,
            pdiff=pdiff,
            pos_effect=pos_effect,
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """RobustKT two-stream architecture with causal smoothing."""

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
        emb_type: str,
        kernel_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.blocks_1 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                    emb_type=emb_type,
                )
                for _ in range(n_blocks)
            ]
        )
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                    emb_type=emb_type,
                )
                for _ in range(n_blocks * 2)
            ]
        )
        self.smooth = Smooth(dropout, d_model, kernel_size)

        # Precompute causal masks to eliminate per-forward numpy->torch allocation
        self.register_buffer(
            "_mask_incl_diag",
            torch.tril(torch.ones(1, 1, max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )
        self.register_buffer(
            "_mask_excl_diag",
            torch.tril(
                torch.ones(1, 1, max_seq_len, max_seq_len, dtype=torch.bool),
                diagonal=-1,
            ),
            persistent=False,
        )
        # Precompute |i-j| position distance matrix
        idx = torch.arange(max_seq_len)
        self.register_buffer(
            "_pos_effect",
            torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)).float()[None, None, :, :],
            persistent=False,
        )

    def forward(
        self,
        q_embed_data: torch.Tensor,
        qa_embed_data: torch.Tensor,
        pid_embed_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.smooth(qa_embed_data)
        x = self.smooth(q_embed_data)

        seq_len = q_embed_data.size(1)
        mask_incl = self._mask_incl_diag[:, :, :seq_len, :seq_len]
        mask_excl = self._mask_excl_diag[:, :, :seq_len, :seq_len]
        pos_effect = self._pos_effect[:, :, :seq_len, :seq_len]

        for block in self.blocks_1:
            y = block(
                causal_mask=mask_incl,
                query=y,
                key=y,
                values=y,
                zero_pad=False,
                pos_effect=pos_effect,
                pdiff=pid_embed_data,
            )

        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(
                    causal_mask=mask_incl,
                    query=x,
                    key=x,
                    values=x,
                    zero_pad=False,
                    apply_pos=False,
                    pos_effect=pos_effect,
                    pdiff=pid_embed_data,
                )
                flag_first = False
            else:
                x = block(
                    causal_mask=mask_excl,
                    query=x,
                    key=x,
                    values=y,
                    zero_pad=True,
                    apply_pos=True,
                    pos_effect=pos_effect,
                    pdiff=pid_embed_data,
                )
                flag_first = True
        return x


class RobustKT(nn.Module):
    """RobustKT knowledge tracing model."""

    def __init__(
        self,
        *,
        num_skills: int,
        num_questions: int,
        dropout: float,
        kq_same: int,
        l2: float,
        separate_qa: int,
        d_model: int,
        n_blocks: int,
        num_attn_heads: int,
        d_ff: int,
        final_fc_dim: int,
        kernel_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.model_name = "robustkt"
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.n_pid = self.num_questions
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = bool(separate_qa)
        self.emb_type = "qid"
        embed_l = d_model

        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions + 1, 1)
            self.q_embed_diff = nn.Embedding(self.num_skills + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)

        self.q_embed = nn.Embedding(self.num_skills, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.num_skills + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_ff=d_ff,
            kq_same=kq_same,
            emb_type=self.emb_type,
            kernel_size=kernel_size,
            max_seq_len=max_seq_len,
        )
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        self.reset()

    def reset(self) -> None:
        if self.num_questions > 0:
            torch.nn.init.constant_(self.difficult_param.weight, 0.0)

    def build_pid_data(
        self, question: torch.Tensor | None, valid_mask: torch.Tensor
    ) -> torch.Tensor | None:
        """Build Rasch pid data with 0 reserved for padding."""
        if self.num_questions <= 0 or question is None:
            return None
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def base_emb(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_embed_data = self.q_embed(sequence)
        if self.separate_qa:
            qa_data = sequence + self.num_skills * response
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(response) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor | None = None,
        question: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_embed_data, qa_embed_data = self.base_emb(sequence, response)
        c_reg_loss = torch.tensor(0.0, device=sequence.device)
        pid_embed_data = None
        valid_mask = (
            torch.ones_like(sequence, dtype=torch.bool) if mask is None else mask
        )
        pid_data = self.build_pid_data(question, valid_mask)

        if self.num_questions > 0 and pid_data is not None:
            q_embed_diff_data = self.q_embed_diff(sequence)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(response)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )
            c_reg_loss = (pid_embed_data**2).sum() * self.l2

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        return torch.sigmoid(output), c_reg_loss
