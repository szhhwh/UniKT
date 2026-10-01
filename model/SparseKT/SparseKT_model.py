"""SparseKT 模型实现"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def attention(q, k, v, d_k, mask, dropout, zero_pad, emb_type, sparse_ratio, k_index):
    """稀疏注意力计算

    Args:
        q: 查询张量，形状为 [batch, heads, seq_len, d_k]
        k: 键张量，形状为 [batch, heads, seq_len, d_k]
        v: 值张量，形状为 [batch, heads, seq_len, d_k]
        d_k: 每个头的维度
        mask: 因果注意力掩码（上三角为 0）
        dropout: Dropout 层
        zero_pad: 是否将第一行注意力分数置零（第一题无历史交互）
        emb_type: 嵌入类型字符串，决定稀疏化策略：
            - 含 "sparseattn"：top-k 稀疏注意力（保留每行前 k_index 个最大分数）
            - 含 "accumulative" 且 sparse_ratio < 1：累积和阈值稀疏化
            - 其它：标准稠密注意力
        sparse_ratio: 累积稀疏化的累计阈值（仅 accumulative 模式使用）
        k_index: top-k 稀疏注意力保留的分数数量

    Returns:
        (output, attn_weights): output 形状 [batch, heads, seq_len, d_k]，
        attn_weights 为 dropout 前的注意力权重（用于可视化/调试）。
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.shape[:3]

    # 屏蔽未来位置
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if "sparseattn" in emb_type:
        # Top-k 稀疏注意力：query 位置 > k_index 时，仅保留 top-k_index 个分数
        if k_index + 1 >= seqlen:
            # 序列过短，无需稀疏化
            pass
        else:
            # 前 k_index+1 行（query 位置）保持不变（历史不足以稀疏）
            scores_a = scores[:, :, : k_index + 1, :]
            # 其余行展平后做 top-k 筛选
            scores_b = scores[:, :, k_index + 1 :, :].reshape(
                bs * head * (seqlen - k_index - 1), -1
            )
            topk_vals, _ = torch.topk(scores_b, k_index, dim=-1)
            scores_t = topk_vals[:, -1:]
            # 保留 scores >= 阈值，其余置 -1e32。
            scores_b = scores_b.masked_fill(scores_b - scores_t < 0, -1e32).reshape(
                bs, head, seqlen - k_index - 1, -1
            )
            # 对保留下来的分数重新 softmax
            scores_b = F.softmax(scores_b, dim=-1)
            scores = torch.cat([scores_a, scores_b], dim=2)
        attn_weights = scores

    elif "accumulative" in emb_type and sparse_ratio < 1.0:
        # 累积稀疏注意力：保留累计概率达到 sparse_ratio 的最小数量分数
        scores = torch.reshape(scores, (bs * head * seqlen, -1))
        sorted_scores, _ = torch.sort(scores, descending=True)
        acc_scores = torch.cumsum(sorted_scores, dim=1)
        # 找到累积和首次 >= sparse_ratio 的位置，保留该位置及之前（更大）的分数
        acc_scores_b = (acc_scores >= sparse_ratio).long()
        idx = torch.argmax(acc_scores_b, dim=1, keepdim=True)
        idx_matrix = torch.arange(seqlen, device=scores.device)[None, :]
        # 排序后下标 <= idx 的位置保留（即 top-(idx+1) 个）
        new_mask = torch.where(idx_matrix <= idx, 0, 1).float()
        sorted_scores = new_mask * sorted_scores
        # 用 -1 标记被屏蔽的分数，便于后续通过最大值阈值还原。
        # masked_fill 与原 where(==0, full(-1), x) 数值等价，免去分配常量张量。
        sorted_scores = sorted_scores.masked_fill(sorted_scores == 0.0, -1.0)
        tmp_scores, _ = torch.max(sorted_scores, dim=1)
        tmp_scores = tmp_scores.unsqueeze(-1)
        # 原始 scores 中大于保留阈值的位置保持，否则置 -1e32
        new_scores = scores.masked_fill(tmp_scores - scores >= 0, -1e32).reshape(
            (bs, head, seqlen, -1)
        )
        new_scores = new_scores.masked_fill(new_scores == 0, -1e32)
        scores = F.softmax(new_scores, dim=-1)
        attn_weights = scores

    else:
        # 标准稠密注意力
        attn_weights = scores

    # 第一行注意力分数置零：第一道题无历史交互信息

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头稀疏注意力层"""

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

    def forward(self, q, k, v, mask, zero_pad, emb_type, sparse_ratio, k_index):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores, attn_weights = attention(
            q,
            k,
            v,
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            emb_type,
            sparse_ratio=sparse_ratio,
            k_index=k_index,
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)

        return output, attn_weights


class TransformerLayer(nn.Module):
    """Transformer 层（因果注意力 + FFN + 残差 + LayerNorm）"""

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1

        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self._causal_mask_cache: dict = {}

    def _build_causal_mask(self, seqlen: int, mask: int, device) -> torch.Tensor:
        """构建因果注意力掩码。

        ``mask=0`` 时允许 query i 注意到位置 j < i（严格下三角）。

        返回:
            形状 [1, 1, seqlen, seqlen] 的布尔张量，True 表示允许注意（保留），
            False 表示屏蔽（在 attention 中被填 -1e32）。
        """
        key = (seqlen, mask, device)
        src_mask = self._causal_mask_cache.get(key)
        if src_mask is None:
            ones = torch.ones((seqlen, seqlen), dtype=torch.bool, device=device)
            src_mask = torch.tril(ones, diagonal=mask - 1).view(1, 1, seqlen, seqlen)
            self._causal_mask_cache[key] = src_mask
        return src_mask

    def forward(
        self,
        mask,
        query,
        key,
        values,
        apply_pos=True,
        emb_type="qid",
        sparse_ratio=0.8,
        k_index=5,
    ):
        seqlen = query.size(1)
        src_mask = self._build_causal_mask(seqlen, mask, query.device)

        if mask == 0:
            # mask=0：当前 query 不能看到当前位置的 response，且第一行置零
            query2, _ = self.masked_attn_head(
                query,
                key,
                values,
                mask=src_mask,
                zero_pad=True,
                emb_type=emb_type,
                sparse_ratio=sparse_ratio,
                k_index=k_index,
            )
        else:
            query2, _ = self.masked_attn_head(
                query,
                key,
                values,
                mask=src_mask,
                zero_pad=False,
                emb_type=emb_type,
                sparse_ratio=sparse_ratio,
                k_index=k_index,
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query, _


class CosinePositionalEmbedding(nn.Module):
    """余弦位置编码"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(1), :]


class Architecture(nn.Module):
    """SparseKT 的 Transformer 架构"""

    def __init__(
        self,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        seq_len,
        emb_type,
        sparse_ratio,
        k_index,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        self.sparse_ratio = sparse_ratio
        self.k_index = k_index

        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(n_blocks)
            ]
        )
        self.position_emb = CosinePositionalEmbedding(
            d_model=self.d_model, max_len=seq_len
        )

    def forward(self, q_embed_data, qa_embed_data):
        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        y = qa_embed_data
        x = q_embed_data

        attn_weights = None
        for block in self.blocks_2:
            x, attn_weights = block(
                mask=0,
                query=x,
                key=x,
                values=y,
                apply_pos=True,
                emb_type=self.emb_type,
                sparse_ratio=self.sparse_ratio,
                k_index=self.k_index,
            )
        return x, attn_weights


class SparseKT(nn.Module):
    """SparseKT 模型

    - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
    - 第一个位置：y_hat[:, 0] 基于 sequence[0:1] 和空历史预测 response[0]

    Args:
        num_skills: 技能（概念）数量
        n_pid: Problem ID 数量（>0 时启用 Rasch 难度调节）
        d_model: 模型维度
        n_blocks: Transformer 块数量
        dropout: Dropout 概率
        d_ff: 前馈网络维度
        n_heads: 注意力头数量
        seq_len: 最大序列长度
        kq_same: 是否共享 key 和 query 的权重
        separate_qa: 是否使用独立的交互嵌入
        final_fc_dim: 第一层全连接层维度
        final_fc_dim2: 第二层全连接层维度
        emb_type: 嵌入/注意力类型，决定稀疏化策略：
            - "qid_sparseattn"（默认）：top-k 稀疏注意力
            - "qid_accumulative"：累积和阈值稀疏注意力
            - "qid"：标准稠密注意力
        sparse_ratio: 累积稀疏化的累计阈值（仅 accumulative 模式使用）
        k_index: top-k 稀疏注意力保留的分数数量
    """

    def __init__(
        self,
        num_skills: int,
        n_pid: int,
        d_model: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.1,
        d_ff: int = 256,
        n_heads: int = 8,
        seq_len: int = 200,
        kq_same: int = 1,
        separate_qa: bool = False,
        final_fc_dim: int = 512,
        final_fc_dim2: int = 256,
        emb_type: str = "qid_sparseattn",
        sparse_ratio: float = 0.8,
        k_index: int = 5,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.sparse_ratio = sparse_ratio
        self.k_index = k_index
        embed_l = d_model

        # Problem ID 相关嵌入
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_skills + 1, embed_l)

        # 技能嵌入层
        self.q_embed = nn.Embedding(num_skills, embed_l)

        # 交互嵌入层
        if separate_qa:
            self.qa_embed = nn.Embedding(2 * num_skills + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        # Transformer 架构
        self.model = Architecture(
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model // n_heads,
            d_ff=d_ff,
            kq_same=kq_same,
            seq_len=seq_len,
            emb_type=emb_type,
            sparse_ratio=sparse_ratio,
            k_index=k_index,
        )

        # 输出层
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        self.reset()

    def reset(self):
        """将 Rasch 难度嵌入初始化为 0（题目难度初始为中性）"""
        if self.n_pid > 0:
            for p in self.parameters():
                if p.size(0) == self.n_pid + 1:
                    torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        """基础嵌入

        Args:
            q_data: 技能ID序列
            target: 响应序列

        Returns:
            技能嵌入和交互嵌入
        """
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        pid_data: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播

        Args:
            sequence: 技能ID序列，形状为 [batch_size, seq_len]
            response: 响应序列，形状为 [batch_size, seq_len]
            mask: 有效位置掩码，形状为 [batch_size, seq_len]
            pid_data: Problem ID 序列，形状为 [batch_size, seq_len]（0 表示填充）

        Returns:
            preds: 预测结果，形状为 [batch_size, seq_len]
        """
        target = response

        # 获取基础嵌入
        q_embed_data, qa_embed_data = self.base_emb(sequence, target)

        # Problem ID 嵌入和 Rasch 难度调节
        if self.n_pid > 0:
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_diff_data = self.q_embed_diff(sequence)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

        # 通过稀疏注意力 Transformer
        d_output, _ = self.model(q_embed_data, qa_embed_data)

        # 拼接输出和技能嵌入
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        preds = torch.sigmoid(output)
        return preds
