"""SimpleKT 模型实现"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """注意力计算函数

    Args:
        q: 查询张量，形状为 [batch, heads, seq_len, d_k]
        k: 键张量，形状为 [batch, heads, seq_len, d_k]
        v: 值张量，形状为 [batch, heads, seq_len, d_k]
        d_k: 每个头的维度
        mask: 注意力掩码
        dropout: Dropout 层
        zero_pad: 是否对第一行进行零填充

    Returns:
        注意力输出，形状为 [batch, heads, seq_len, d_k]
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    """多头注意力层"""

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

    def forward(self, q, k, v, mask, zero_pad):
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

        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)

        return output


class TransformerLayer(nn.Module):
    """Transformer 层"""

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
        src_mask = self._causal_mask_k1 if mask else self._causal_mask_k0
        if (
            src_mask is None
            or seqlen != self._const_seqlen
            or src_mask.device != query.device
        ):
            self._build_constants(seqlen, query.device)
            src_mask = self._causal_mask_k1 if mask else self._causal_mask_k0

        query2 = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=mask == 0
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query


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
    """SimpleKT 的 Transformer 架构"""

    def __init__(
        self, n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same, seq_len
    ):
        super().__init__()
        self.d_model = d_model

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

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)

        return x


class SimpleKT(nn.Module):
    """SimpleKT 模型

    基于 Transformer 的知识追踪模型，使用技能序列和响应序列进行预测。

    Args:
        num_skills: 技能（概念）数量
        n_pid: Problem ID数量
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
    """

    def __init__(
        self,
        num_skills: int,
        n_pid: int,
        d_model: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.1,
        d_ff: int = 256,
        n_heads: int = 4,
        seq_len: int = 200,
        kq_same: int = 1,
        separate_qa: bool = False,
        final_fc_dim: int = 512,
        final_fc_dim2: int = 256,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        embed_l = d_model

        # Problem ID相关嵌入（Rasch模型）
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        SimpleKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 第一个位置：y_hat[:, 0] 基于 sequence[0:1] 和空历史预测 response[0]

        Args:
            sequence: 技能ID序列，形状为 [batch_size, seq_len]
            response: 响应序列，形状为 [batch_size, seq_len]
            mask: 有效位置掩码，形状为 [batch_size, seq_len]
            pid_data: Problem ID序列，形状为 [batch_size, seq_len]

        Returns:
            preds: 预测结果，形状为 [batch_size, seq_len]
        """
        target = response

        # 获取基础嵌入
        q_embed_data, qa_embed_data = self.base_emb(sequence, target)

        # Problem ID嵌入和Rasch难度调节
        if self.n_pid > 0:
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_diff_data = self.q_embed_diff(sequence)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

        # 通过 Transformer
        d_output = self.model(q_embed_data, qa_embed_data)

        # 拼接输出和技能嵌入
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        # Sigmoid 激活
        preds = torch.sigmoid(output)

        return preds
