"""extraKT 模型实现

基于ALiBi (Attention with Linear Biases) 位置编码的双流Transformer知识追踪模型。
与AKT架构相似，但使用ALiBi替代单调距离衰减注意力。

核心特点：
- ALiBi位置编码：通过线性偏置替代传统位置嵌入
- 双流Transformer：blocks_1 (QA编码器) + blocks_2 (知识检索器)
- Rasch模型：题目难度参数建模
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def _get_slopes(n: int) -> list[float]:
    """计算ALiBi的注意力头斜率。

    对于2的幂次方头数，使用几何级数；否则回退到最近的2的幂次方。

    Args:
        n: 注意力头数量

    Returns:
        长度为n的斜率列表
    """

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + _get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def attention(q, k, v, d_k, mask, dropout, zero_pad, alibi=None):
    """带ALiBi偏置的注意力计算函数。

    Args:
        q: Query张量 [BS, n_heads, seq_len, d_k]
        k: Key张量 [BS, n_heads, seq_len, d_k]
        v: Value张量 [BS, n_heads, seq_len, d_k]
        d_k: 每个头的维度
        mask: 因果掩码 (bool, True=可参与)
        dropout: Dropout层
        zero_pad: 是否对第一行进行零填充
        alibi: ALiBi偏置矩阵 [1, n_heads, maxpos, maxpos]

    Returns:
        output: 注意力输出 [BS, seq_len, d_model]
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 添加ALiBi偏置
    if alibi is not None:
        seq_len = scores.size(-1)
        scores = scores + alibi[:, :, :seq_len, :seq_len]

    scores.masked_fill_(~mask, -1e32)
    scores = F.softmax(scores, dim=-1)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    """带ALiBi位置偏置的多头注意力机制"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self._reset_parameters()

        # ALiBi位置偏置
        maxpos = 1000
        slopes = torch.tensor(_get_slopes(n_heads)) * -1
        context_position = torch.arange(maxpos)[:, None]
        memory_position = torch.arange(maxpos)[None, :]
        relative_position = memory_position - context_position
        relative_position = (
            torch.abs(relative_position).unsqueeze(0).expand(n_heads, -1, -1)
        )
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        self.register_buffer("alibi", alibi.unsqueeze(0), persistent=False)

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

        scores = attention(
            q, k, v, self.d_k, mask, self.dropout, zero_pad, alibi=self.alibi
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


class TransformerLayer(nn.Module):
    """Transformer层，包含多头注意力和前馈网络"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
    ):
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
        self._mask_seqlen = 0

    def _build_causal_masks(self, seqlen: int, device: torch.device) -> None:
        ones = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device)
        self._causal_mask_k1 = ones.tril().view(1, 1, seqlen, seqlen)
        self._causal_mask_k0 = ones.tril(diagonal=-1).view(1, 1, seqlen, seqlen)
        self._mask_seqlen = seqlen

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen = query.size(1)
        cm = self._causal_mask_k1
        if cm is None or seqlen != self._mask_seqlen or cm.device != query.device:
            self._build_causal_masks(seqlen, query.device)
        src_mask = self._causal_mask_k1 if mask else self._causal_mask_k0

        query2 = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=not mask
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """extraKT架构，包含QA编码器和知识检索器"""

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
    ):
        super().__init__()

        # QA编码器：编码历史交互信息
        self.blocks_1 = nn.ModuleList(
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
        # 知识检索器：从历史中检索相关知识状态
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
                for _ in range(n_blocks * 2)
            ]
        )

    def forward(self, q_embed_data, qa_embed_data):
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        # Encoder: 对0~t-1时刻前的qa信息进行编码
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)

        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # Peek current question
                x = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    apply_pos=False,
                )
                flag_first = False
            else:  # Don't peek current response
                x = block(
                    mask=0,
                    query=x,
                    key=x,
                    values=y,
                    apply_pos=True,
                )
                flag_first = True
        return x


class extraKT(nn.Module):
    """extraKT 模型

    基于ALiBi位置编码的双流Transformer知识追踪模型：
    - blocks_1: QA编码器，编码历史交互信息
    - blocks_2: 知识检索器，从历史中检索相关知识状态
    - ALiBi: 通过线性偏置替代传统位置嵌入，提供位置感知能力
    - Rasch模型: 通过题目难度参数建模题目差异

    Args:
        num_c: 概念（技能）数量
        n_pid: Problem ID数量（题目数量），0表示不使用Problem ID
        d_model: 模型隐藏维度
        n_blocks: Transformer块数量
        dropout: Dropout概率
        d_ff: 前馈网络维度
        kq_same: Key和Query是否使用相同的线性变换
        final_fc_dim: 最终全连接层维度
        num_attn_heads: 注意力头数量
        separate_qa: 是否使用独立的QA嵌入
        l2: L2正则化系数（用于Rasch模型）
        emb_type: 嵌入类型
        num_buckets: ALiBi桶数量（当前ALiBi路径下未使用）
        max_distance: ALiBi最大距离（当前ALiBi路径下未使用）
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        d_model: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.05,
        d_ff: int = 256,
        kq_same: int = 1,
        final_fc_dim: int = 512,
        num_attn_heads: int = 8,
        separate_qa: bool = False,
        l2: float = 1e-5,
        emb_type: str = "qid",
        num_buckets: int = 32,
        max_distance: int = 100,
    ):
        super().__init__()
        self.model_name = "extraKT"
        self.num_c = num_c
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Problem ID相关嵌入（Rasch模型）
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)  # 题目难度
            self.q_embed_diff = nn.Embedding(
                self.num_c + 1, embed_l
            )  # question差异向量
            self.qa_embed_diff = nn.Embedding(
                2 * self.num_c + 1, embed_l
            )  # interaction差异向量

        # 概念嵌入层
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.num_c, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.num_c + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture: 双流Transformer
        self.model = Architecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=self.kq_same,
        )

        # 输出层
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        # 初始化Rasch模型参数
        self.reset()

    def reset(self):
        """初始化Rasch模型参数为0"""
        if self.n_pid > 0:
            for p in self.parameters():
                if p.size(0) == self.n_pid + 1:
                    torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        """计算基础嵌入

        Args:
            q_data: 问题ID（技能ID）序列
            target: 响应序列

        Returns:
            q_embed_data: 问题嵌入
            qa_embed_data: 问题-响应交互嵌入
        """
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.num_c * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor = None,
        pid_data: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            sequence: 概念ID序列，形状为 [batch_size, sequence_length]
            response: 响应序列，形状为 [batch_size, sequence_length]
            mask: 有效位置掩码，形状为 [batch_size, sequence_length]
            pid_data: Problem ID序列，形状为 [batch_size, sequence_length]

        Returns:
            preds: 预测结果，形状为 [batch_size, sequence_length]
            c_reg_loss: Rasch模型正则化损失
        """
        # 获取基础嵌入
        q_embed_data, qa_embed_data = self.base_emb(sequence, response)

        # 处理Problem ID和Rasch模型
        pid_embed_data = None
        c_reg_loss = torch.tensor(0.0, device=sequence.device)

        if self.n_pid > 0 and pid_data is not None:
            # 计算问题差异向量嵌入
            q_embed_diff_data = self.q_embed_diff(sequence)
            pid_embed_data = self.difficult_param(pid_data)

            # question encoder: uq * d_ct + c_ct
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            # 计算交互差异向量嵌入
            qa_embed_diff_data = self.qa_embed_diff(response)

            if self.separate_qa:
                # uq * f_(ct,rt) + e_(ct,rt)
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                # uq * (h_rt + d_ct)
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )

            # Rasch模型正则化损失
            c_reg_loss = (pid_embed_data**2).sum() * self.l2

        # 通过双流Transformer架构
        d_output = self.model(q_embed_data, qa_embed_data)

        # 拼接输出和问题嵌入
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        # Sigmoid激活
        preds = torch.sigmoid(output)

        # extraKT输出语义说明：
        # extraKT的第二个流（blocks_2）第一层可以"peek current question"
        # 因此 preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]

        return preds, c_reg_loss
