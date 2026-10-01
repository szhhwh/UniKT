"""StableKT 模型实现

StableKT: Enhancing Length Generalization for Attention Based Knowledge Tracing
Models with Linear Biases.

该模型结合了标准注意力（ALiBi）和半影锥注意力（HAKT），使用一半注意力头进行标准注意力计算，
另一半进行半影锥注意力计算，以提升知识追踪模型在长序列上的泛化能力。

支持多种位置编码方式：ALiBi（默认）、T5 相对位置偏置、旋转位置编码（RoPE）、正弦/余弦位置编码。
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def _map_psi(x, r):
    """将输入映射到双曲空间的半影锥表示

    将最后一维拆分为空间坐标和高度坐标。

    Args:
        x: 输入张量，形状为 [..., d_k]
        r: 半影锥半径

    Returns:
        空间坐标和高度坐标的元组
    """
    x_x = x[..., :-1]
    x_y = torch.sigmoid(x[..., -1])
    return x_x * x_y.unsqueeze(-1) * r, x_y * r


def _penumbral(q, k, r=1, gamma=0.1, eps=1e-6):
    """计算半影锥注意力分数

    基于双曲空间中的半影锥几何结构计算查询-键对之间的注意力分数。

    Args:
        q: 查询张量，形状为 [batch, heads, seq_len, d_k]
        k: 键张量，形状为 [batch, heads, seq_len, d_k]
        r: 半影锥半径
        gamma: 温度参数
        eps: 数值稳定的小常数

    Returns:
        注意力分数，形状为 [batch, heads, seq_len, seq_len]
    """
    q_x, q_y = _map_psi(q, r)
    k_x, k_y = _map_psi(k, r)
    q_y = q_y.unsqueeze(3)
    k_y = k_y.unsqueeze(2)

    x_q_y = torch.sqrt(r**2 - q_y**2 + eps)
    x_k_y = torch.sqrt(r**2 - k_y**2 + eps)

    pairwise_dist = torch.cdist(q_x, k_x)

    lca_height = torch.maximum(
        torch.maximum(q_y**2, k_y**2),
        r**2 - ((x_q_y + x_k_y - pairwise_dist) / 2) ** 2,
    )

    lca_height_outcone = (
        (pairwise_dist**2 + k_y**2 - q_y**2) / (2 * pairwise_dist + eps)
    ) ** 2 + q_y**2

    exists_cone = torch.logical_or(
        pairwise_dist <= x_q_y,
        (pairwise_dist - x_q_y) ** 2 + k_y**2 <= r**2,
    )

    return -gamma * torch.where(exists_cone, lca_height, lca_height_outcone)


def _attention(
    q, k, v, d_k, mask, dropout, zero_pad, alibi, emb_type, rel_pos_bias, rotary_pe
):
    """标准注意力计算（支持 ALiBi / T5 / RoPE 位置编码）

    Args:
        q: 查询张量
        k: 键张量
        v: 值张量
        d_k: 每个头的维度
        mask: 因果注意力掩码
        dropout: Dropout 层
        zero_pad: 是否对第一行进行零填充
        alibi: ALiBi 偏置缓冲区
        emb_type: 嵌入类型
        rel_pos_bias: T5 相对位置偏置模块
        rotary_pe: 旋转位置编码模块

    Returns:
        注意力输出
    """
    if emb_type.find("rotary") != -1:
        q = rotary_pe(q)
        k = rotary_pe(k)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    seq_len = scores.size(-1)

    if emb_type.find("sin") != -1 or emb_type.find("wha") != -1:
        pass  # position encoding already applied upstream
    elif emb_type.find("t5") != -1:
        scores = scores + rel_pos_bias(scores)
    elif emb_type.find("rotary") == -1:
        scores = scores + alibi[:, :, :seq_len, :seq_len]

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


def _attention_hakt(
    q,
    k,
    v,
    d_k,
    mask,
    dropout,
    zero_pad,
    alibi,
    r,
    gamma,
    emb_type,
    rel_pos_bias,
    rotary_pe,
):
    """半影锥注意力计算（HAKT）

    Args:
        q: 查询张量
        k: 键张量
        v: 值张量
        d_k: 每个头的维度
        mask: 因果注意力掩码
        dropout: Dropout 层
        zero_pad: 是否对第一行进行零填充
        alibi: ALiBi 偏置缓冲区
        r: 半影锥半径
        gamma: 温度参数
        emb_type: 嵌入类型
        rel_pos_bias: T5 相对位置偏置模块
        rotary_pe: 旋转位置编码模块

    Returns:
        注意力输出
    """
    if emb_type.find("rotary") != -1:
        q = rotary_pe(q)
        k = rotary_pe(k)

    scores = _penumbral(q, k, r, gamma) / math.sqrt(d_k)
    seq_len = scores.size(-1)

    if emb_type.find("sin") != -1 or emb_type.find("wha") != -1:
        pass
    elif emb_type.find("t5") != -1:
        scores = scores + rel_pos_bias(scores)
    elif emb_type.find("rotary") == -1:
        scores = scores + alibi[:, :, :seq_len, :seq_len]

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class CosinePositionalEmbedding(nn.Module):
    """余弦位置编码"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        max_len = 1000
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


class SinePositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.register_buffer(
            "pos_table",
            self._get_sinusoid_encoding_table(n_position, d_hid),
            persistent=False,
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        n_position = 1000

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class T5RelativePositionBias(nn.Module):
    """T5 相对位置偏置"""

    def __init__(self, scale, causal=True, num_buckets=16, max_distance=50):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=16, max_distance=50
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = values.squeeze(-1)
        return bias * self.scale


class RotaryPositionalEmbeddings(nn.Module):
    """旋转位置编码（RoPE）"""

    def __init__(self, d, base=10000):
        super().__init__()
        self.theta = nn.Parameter(
            1.0 / (base ** (torch.arange(0, d, 2).float() / d)),
            requires_grad=False,
        )

    def forward(self, x):
        batch_size, seq_len, n_heads, d = x.shape
        d_2 = d // 2

        seq_idx = torch.arange(seq_len, device=x.device).type_as(self.theta)
        idx_theta = torch.einsum("n,d->nd", seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        neg_half_x = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

        rx = (x * idx_theta2.cos()[None, :, None, :]) + (
            neg_half_x * idx_theta2.sin()[None, :, None, :]
        )
        return rx


def _get_slopes(n):
    """计算 ALiBi 的注意力头斜率

    Args:
        n: 注意力头数量

    Returns:
        长度为 n 的斜率列表
    """

    def _get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_slopes_power_of_2(closest_power_of_2)
            + _get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


class MultiHeadAttention(nn.Module):
    """多头注意力层

    支持标准注意力和半影锥注意力的混合模式：
    - 默认模式：前半注意力头使用标准注意力，后半使用半影锥注意力（HAKT）
    - woha 模式（emb_type 含 "woha"）：所有注意力头使用标准注意力
    """

    def __init__(
        self,
        d_model,
        d_feature,
        n_heads,
        dropout,
        kq_same,
        seq_len,
        r,
        gamma,
        emb_type,
        num_buckets,
        max_distance,
        bias=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.r = r
        self.gamma = gamma
        self.emb_type = emb_type

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if emb_type.find("t5") != -1:
            self.rel_pos_bias = T5RelativePositionBias(
                scale=d_model**0.5,
                causal=True,
                num_buckets=num_buckets,
                max_distance=max_distance,
            )
        else:
            self.rel_pos_bias = None

        if emb_type.find("rotary") != -1:
            self.rotary_pe = RotaryPositionalEmbeddings(self.d_k)
        else:
            self.rotary_pe = None

        self._reset_parameters()

        # Register ALiBi bias as a buffer so it follows the model to its device.
        maxpos = 1000
        context_position = torch.arange(maxpos).unsqueeze(1)
        memory_position = torch.arange(maxpos).unsqueeze(0)
        relative_position = (memory_position - context_position).abs()
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)

        slopes = torch.tensor(_get_slopes(n_heads)) * -1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.unsqueeze(0)  # [1, n_heads, maxpos, maxpos]
        self.register_buffer("alibi", alibi, persistent=False)

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
        half_h = self.h // 2

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.emb_type.find("woha") != -1:
            scores = _attention(
                q,
                k,
                v,
                self.d_k,
                mask,
                self.dropout,
                zero_pad,
                alibi=self.alibi,
                emb_type=self.emb_type,
                rel_pos_bias=self.rel_pos_bias,
                rotary_pe=self.rotary_pe,
            )
        else:
            scores = _attention(
                q[:, :half_h],
                k[:, :half_h],
                v[:, :half_h],
                self.d_k,
                mask,
                self.dropout,
                zero_pad,
                alibi=self.alibi[:, :half_h],
                emb_type=self.emb_type,
                rel_pos_bias=self.rel_pos_bias,
                rotary_pe=self.rotary_pe,
            )
            scores_hakt = _attention_hakt(
                q[:, half_h:],
                k[:, half_h:],
                v[:, half_h:],
                self.d_k,
                mask,
                self.dropout,
                zero_pad,
                alibi=self.alibi[:, half_h:],
                r=self.r,
                gamma=self.gamma,
                emb_type=self.emb_type,
                rel_pos_bias=self.rel_pos_bias,
                rotary_pe=self.rotary_pe,
            )
            scores = torch.cat((scores, scores_hakt), dim=1)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


class TransformerLayer(nn.Module):
    """StableKT Transformer 层

    包含多头注意力和前馈网络，支持残差连接和层归一化。
    """

    def __init__(
        self,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        seq_len,
        r,
        gamma,
        emb_type,
        num_buckets,
        max_distance,
    ):
        super().__init__()
        kq_same = kq_same == 1

        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=kq_same,
            seq_len=seq_len,
            r=r,
            gamma=gamma,
            emb_type=emb_type,
            num_buckets=num_buckets,
            max_distance=max_distance,
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
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)

        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True
            )
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query


class Architecture(nn.Module):
    """StableKT 的 Transformer 架构

    由多个 TransformerLayer 堆叠组成，使用交互嵌入作为 value，技能嵌入作为 query/key。
    """

    def __init__(
        self,
        n_blocks,
        d_model,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        seq_len,
        r,
        gamma,
        emb_type,
        num_buckets,
        max_distance,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type

        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    kq_same=kq_same,
                    seq_len=seq_len,
                    r=r,
                    gamma=gamma,
                    emb_type=emb_type,
                    num_buckets=num_buckets,
                    max_distance=max_distance,
                )
                for _ in range(n_blocks)
            ]
        )

        if self.emb_type.find("sin") != -1:
            self.position_emb = SinePositionalEncoding(
                d_hid=self.d_model, n_position=seq_len
            )
        else:
            self.position_emb = CosinePositionalEmbedding(
                d_model=self.d_model, max_len=seq_len
            )

    def forward(self, q_embed_data, qa_embed_data):
        # Add position encoding only for sin and wha modes.
        if self.emb_type.find("sin") != -1 or self.emb_type.find("wha") != -1:
            q_posemb = self.position_emb(q_embed_data)
            q_embed_data = q_embed_data + q_posemb
            qa_posemb = self.position_emb(qa_embed_data)
            qa_embed_data = qa_embed_data + qa_posemb

        y = qa_embed_data
        x = q_embed_data

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)

        return x


class StableKT(nn.Module):
    """StableKT 模型

    基于 Transformer 的知识追踪模型，结合标准注意力（ALiBi）和半影锥注意力（HAKT），
    使用一半注意力头进行标准注意力计算，另一半进行半影锥注意力计算。

    StableKT 预测语义：
    - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
    - 第一个位置：y_hat[:, 0] 基于 sequence[0:1] 和空历史预测 response[0]

    Args:
        num_skills: 技能（概念）数量
        n_pid: Problem ID 数量
        d_model: 模型维度
        n_blocks: Transformer 块数量
        dropout: Dropout 概率
        d_ff: 前馈网络维度
        n_heads: 注意力头数量（必须为偶数）
        seq_len: 最大序列长度
        kq_same: 是否共享 key 和 query 的权重
        separate_qa: 是否使用独立的交互嵌入
        final_fc_dim: 第一层全连接层维度
        final_fc_dim2: 第二层全连接层维度
        emb_type: 嵌入类型，支持 "qid"（默认）、"qid_woha"（无 HAKT）、
                  "qid_sin"、"qid_t5"、"qid_rotary"、"qid_wha" 等变体
        r: 半影锥半径
        gamma: 半影锥温度参数
        num_buckets: T5 相对位置偏置的分桶数
        max_distance: T5 相对位置偏置的最大距离
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
        emb_type: str = "qid",
        r: float = 1.0,
        gamma: float = 1.0,
        num_buckets: int = 32,
        max_distance: int = 100,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        self.r = r
        self.gamma = gamma
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            else:
                self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_skills + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)

        self.q_embed = nn.Embedding(num_skills, embed_l)

        if separate_qa:
            self.qa_embed = nn.Embedding(2 * num_skills + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            kq_same=kq_same,
            seq_len=seq_len,
            r=r,
            gamma=gamma,
            emb_type=emb_type,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )

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
        """初始化 Rasch 模型参数为零"""
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
            技能嵌入和交互嵌入的元组
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
            pid_data: Problem ID 序列，形状为 [batch_size, seq_len]

        Returns:
            preds: 预测结果，形状为 [batch_size, seq_len]
        """
        target = response

        q_embed_data, qa_embed_data = self.base_emb(sequence, target)

        if self.n_pid > 0 and self.emb_type.find("norasch") == -1:
            q_embed_diff_data = self.q_embed_diff(sequence)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            if self.emb_type.find("aktrasch") != -1:
                # aktrasch variant: also modulate the interaction embedding.
                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )

        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        preds = torch.sigmoid(output)

        return preds
