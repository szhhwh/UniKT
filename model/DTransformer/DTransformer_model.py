"""DTransformer (Diagnostic Transformer) 模型实现"""

import math

import torch
import torch.nn.functional as F
from torch import nn

MIN_SEQ_LEN = 5


def attention(q, k, v, mask, gamma=None, maxout=False, need_weights=True):
    if need_weights or q.size(0) <= 64:
        return _attention(q, k, v, mask, gamma, maxout, need_weights)
    # Split independent knowledge batches, keeping full sequence reductions.
    outputs = []
    for start in range(0, q.size(0), 64):
        stop = start + 64
        chunk_mask = (
            mask[start:stop]
            if mask.dim() == q.dim() and mask.size(0) == q.size(0)
            else mask
        )
        output, _ = _attention(
            q[start:stop],
            k[start:stop],
            v[start:stop],
            chunk_mask,
            gamma,
            maxout,
            False,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=0), None


def _attention(q, k, v, mask, gamma, maxout, need_weights):
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    if gamma is not None:
        x1 = torch.arange(seqlen, dtype=torch.float, device=gamma.device).expand(
            seqlen, -1
        )
        x2 = x1.T.contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)
            dist_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            del scores_
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            torch.sub(disttotal_scores, dist_scores, out=dist_scores)
            del disttotal_scores
            dist_scores.mul_(position_effect).clamp_min_(0.0).sqrt_()

        gamma_val = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma_val).exp(), min=1e-5, max=1e5)
        del dist_scores
        scores = scores * total_effect
        del total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)

    if maxout:
        scale = torch.clamp(1.0 / (scores.max(dim=-1, keepdim=True)[0] + 1e-8), max=5.0)
        scores *= scale

    output = torch.matmul(scores, v)
    return output, scores if need_weights else None


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, maxout=False, need_weights=True):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        v_, scores = attention(q, k, v, mask, self.gammas, maxout, need_weights)

        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output, scores


class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, values, peek_cur=False, need_weights=True):
        seqlen = query.size(1)
        mask = (
            torch.ones(seqlen, seqlen, device=query.device)
            .tril(0 if peek_cur else -1)
            .bool()[None, None, :, :]
        )

        query_, scores = self.masked_attn_head(
            query, key, values, mask, maxout=not peek_cur, need_weights=need_weights
        )
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class DTransformer(nn.Module):
    """DTransformer (Diagnostic Transformer) 模型

    使用可学习知识参数和多头注意力机制进行知识状态诊断。

    Args:
        num_c: 概念（技能）数量
        n_pid: Problem ID数量，0表示不使用Rasch模型
        d_model: 模型隐藏维度
        d_ff: 前馈网络维度
        num_attn_heads: 注意力头数量
        n_know: 知识参数数量
        n_blocks: Transformer块数量（1-3）
        dropout: Dropout概率
        separate_qa: 是否使用独立的QA嵌入
        l2: Rasch模型正则化系数
        shortcut: 使用类AKT模式（跳过知识参数）
        proj: 使用CL投影层
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        d_model: int = 128,
        d_ff: int = 256,
        num_attn_heads: int = 8,
        n_know: int = 16,
        n_blocks: int = 3,
        dropout: float = 0.3,
        separate_qa: bool = False,
        l2: float = 1e-5,
        shortcut: bool = False,
        proj: bool = False,
    ):
        super().__init__()
        self.model_name = "dtransformer"
        self.num_c = num_c
        self.n_pid = n_pid
        self.dropout_rate = dropout
        self.l2 = l2
        self.shortcut = shortcut
        self.n_blocks = n_blocks
        self.separate_qa = separate_qa
        self.n_heads = num_attn_heads
        embed_l = d_model

        # Embeddings
        if self.n_pid > 0:
            self.q_diff_embed = nn.Embedding(num_c + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        self.q_embed = nn.Embedding(num_c, embed_l)
        if self.separate_qa:
            self.s_embed = nn.Embedding(2 * num_c + 1, embed_l)
        else:
            self.s_embed = nn.Embedding(2, embed_l)

        # Transformer layers
        self.block1 = DTransformerLayer(d_model, self.n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, self.n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, self.n_heads, dropout)
        self.block4 = DTransformerLayer(d_model, self.n_heads, dropout, kq_same=False)

        # Knowledge parameters
        self.n_know = n_know
        self.know_params = nn.Parameter(torch.empty(n_know, d_model))
        nn.init.uniform_(self.know_params, -1.0, 1.0)

        # Output layer
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

        # CL projection (optional)
        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        self.reset()

    def reset(self):
        if self.n_pid > 0:
            for p in self.parameters():
                if p.size(0) == self.n_pid + 1:
                    nn.init.constant_(p, 0.0)

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor | None = None,
        pid_data: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """标准前向传播

        Returns:
            preds: 预测概率 [batch_size, seq_len]
            reg_loss: Rasch模型正则化损失
        """
        output, _, _, _, reg_loss, _ = self._predict(
            sequence, response, pid_data, need_weights=False
        )
        preds = torch.sigmoid(output)
        return preds, reg_loss

    def predict(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        pid_data: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """带隐状态的前向传播，用于对比学习

        Returns:
            output: 原始logits [batch_size, seq_len]
            z: 知识状态表示 [batch_size, seq_len, n_know * d_model]
            q_emb: 问题嵌入 [batch_size, seq_len, d_model]
            reg_loss: 正则化损失
        """
        output, _, z, q_emb, reg_loss, _ = self._predict(
            sequence, response, pid_data, need_weights=False
        )
        return output, z, q_emb, reg_loss

    def _base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.num_c * target
            qa_embed_data = self.s_embed(qa_data)
        else:
            qa_embed_data = self.s_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def _embedding(self, q_data, target, pid_data=None):
        q_embed_data, qa_embed_data = self._base_emb(q_data, target)

        p_diff = None
        if self.n_pid > 0 and pid_data is not None:
            q_embed_diff = self.q_diff_embed(q_data)
            p_diff = self.p_diff_embed(pid_data)
            q_embed_data = q_embed_data + p_diff * q_embed_diff

            qa_embed_diff = self.s_diff_embed(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + p_diff * qa_embed_diff
            else:
                qa_embed_data = qa_embed_data + p_diff * (qa_embed_diff + q_embed_diff)

        return q_embed_data, qa_embed_data, p_diff

    def _encode(self, q_emb, s_emb, need_weights=True):
        if self.shortcut:
            hq, _ = self.block1(q_emb, q_emb, q_emb, peek_cur=True)
            hs, scores = self.block2(s_emb, s_emb, s_emb, peek_cur=True)
            return self.block3(hq, hq, hs, peek_cur=False), scores, None

        if self.n_blocks == 1:
            hq = q_emb
            p, q_scores = self.block1(
                q_emb, q_emb, s_emb, peek_cur=True, need_weights=need_weights
            )
        elif self.n_blocks == 2:
            hq = q_emb
            hs, _ = self.block1(s_emb, s_emb, s_emb, peek_cur=True)
            p, q_scores = self.block2(
                hq, hq, hs, peek_cur=True, need_weights=need_weights
            )
        else:
            hq, _ = self.block1(q_emb, q_emb, q_emb, peek_cur=True)
            hs, _ = self.block2(s_emb, s_emb, s_emb, peek_cur=True)
            p, q_scores = self.block3(
                hq, hq, hs, peek_cur=True, need_weights=need_weights
            )

        bs, seqlen, d_model = p.size()
        n_know = self.n_know

        query = (
            self.know_params[None, :, None, :]
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        z, k_scores = self.block4(
            query, hq, p, peek_cur=False, need_weights=need_weights
        )
        z = (
            z.view(bs, n_know, seqlen, d_model)
            .transpose(1, 2)
            .contiguous()
            .view(bs, seqlen, -1)
        )
        if need_weights:
            k_scores = (
                k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

        return z, q_scores, k_scores

    def _readout(self, z, query):
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )
        value = z.reshape(bs * seqlen, self.n_know, -1)

        beta = torch.matmul(key, query.reshape(bs * seqlen, -1, 1)).view(
            bs * seqlen, 1, self.n_know
        )
        alpha = torch.softmax(beta, -1)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)

    def _predict(self, q, s, pid=None, need_weights=True):
        q_emb, s_emb, p_diff = self._embedding(q, s, pid)
        z, q_scores, k_scores = self._encode(q_emb, s_emb, need_weights)

        h = z if self.shortcut else self._readout(z, q_emb)

        concat_q = torch.cat([q_emb, h], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        if p_diff is not None:
            reg_loss = (p_diff**2).mean() * self.l2
        else:
            reg_loss = torch.tensor(0.0, device=q.device)

        return output, concat_q, z, q_emb, reg_loss, (q_scores, k_scores)

    def sim(self, z1, z2):
        """计算对比学习相似度矩阵

        Args:
            z1: 知识状态 [batch_size, seq_len, n_know * d_model]
            z2: 知识状态 [batch_size, seq_len, n_know * d_model]

        Returns:
            相似度矩阵 [batch_size, batch_size, seq_len]
        """
        bs, seqlen, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.n_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.n_know, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05
