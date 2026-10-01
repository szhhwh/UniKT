import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class TimeGapEmbedding(nn.Module):
    """Time gap embedding via one-hot + single linear projection.

    Concatenates one-hot vectors for rgap, sgap, pcount, then projects
    through a single Linear (bias=False). This preserves cross-feature
    interactions that separate Embedding + sum cannot capture.
    """

    def __init__(self, num_rgap: int, num_sgap: int, num_pcount: int, d_model: int):
        super().__init__()
        self.num_rgap = num_rgap
        self.num_sgap = num_sgap
        self.num_pcount = num_pcount
        input_size = num_rgap + num_sgap + num_pcount
        self.time_emb = nn.Linear(input_size, d_model, bias=False)

    def forward(
        self,
        rgap: torch.Tensor,
        sgap: torch.Tensor,
        pcount: torch.Tensor,
    ) -> torch.Tensor:
        rgap_oh = F.one_hot(rgap.clamp(0, self.num_rgap - 1), self.num_rgap).float()
        sgap_oh = F.one_hot(sgap.clamp(0, self.num_sgap - 1), self.num_sgap).float()
        pcount_oh = F.one_hot(
            pcount.clamp(0, self.num_pcount - 1), self.num_pcount
        ).float()
        tg = torch.cat([rgap_oh, sgap_oh, pcount_oh], dim=-1)
        return self.time_emb(tg)


class CosinePositionalEmbedding(nn.Module):
    """固定余弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]


class CausalConv1d(nn.Conv1d):
    """因果卷积 — 左填充保证不窥见未来信息"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs
    ):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=self.__padding,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = super().forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class CIC(nn.Module):
    """Causal Inception Convolution 模块

    两个不同 kernel_size 的因果卷积分支进行交叉门控。
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        kernel_size1: int = 1,
        kernel_size2: int = 3,
        drop: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_features, hidden_features, kernel_size1)
        self.conv2 = CausalConv1d(in_features, hidden_features, kernel_size2)
        self.conv3 = CausalConv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, F, S)
        x1 = self.conv1(x)
        x1_act = self.drop(self.act(x1))
        x2 = self.conv2(x)
        x2_act = self.drop(self.act(x2))
        out = self.conv3(x1 * x2_act + x2 * x1_act)
        return out.transpose(1, 2)  # (B, S, F)


def _get_slopes(n_heads: int) -> list[float]:
    """计算 ALiBi 的斜率"""

    def _slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return _slopes_power_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return (
        _slopes_power_of_2(closest)
        + _get_slopes(2 * closest)[0::2][: n_heads - closest]
    )


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_k: int,
    mask: torch.Tensor,
    dropout: nn.Dropout,
    zero_pad: bool,
    alibi_bias: torch.Tensor,
) -> torch.Tensor:
    """带 ALiBi 偏置的缩放点积注意力"""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    seqlen = scores.size(2)

    # ALiBi: 基于相对距离的线性偏置
    scores = scores + alibi_bias[:, :, :seqlen, :seqlen]
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    # 零填充第一行: 位置 t 不应 attend to 自身的 response

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    """多头注意力 (带 ALiBi 线性偏置)"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        bias: bool = True,
        max_seq_len: int = 200,
    ):
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
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # ALiBi bias (pre-computed, sized to match actual sequence length)
        maxpos = max_seq_len
        context_pos = torch.arange(maxpos)[:, None]
        memory_pos = torch.arange(maxpos)[None, :]
        relative_pos = (
            torch.abs(memory_pos - context_pos).unsqueeze(0).expand(n_heads, -1, -1)
        )
        slopes = torch.tensor(_get_slopes(n_heads)) * -1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_pos
        self.register_buffer(
            "alibi_bias", alibi.view(1, n_heads, maxpos, maxpos), persistent=False
        )

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)
        if self.k_linear.bias is not None:
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
    ) -> torch.Tensor:
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

        scores = attention(
            q, k, v, self.d_k, mask, self.dropout, zero_pad, self.alibi_bias
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    """Transformer 层: MultiHeadAttention + CIC + LayerNorm + Dropout"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        k1: int,
        k2: int,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=kq_same,
            max_seq_len=max_seq_len,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cic = CIC(d_model, d_ff, kernel_size1=k1, kernel_size2=k2, drop=dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        mask: int,
        query: torch.Tensor,
        key: torch.Tensor,
        values: torch.Tensor,
        apply_pos: bool = True,
    ) -> torch.Tensor:
        device = query.device
        seqlen = query.size(1)

        src_mask = torch.ones(1, 1, seqlen, seqlen, dtype=torch.bool, device=device)
        src_mask = torch.triu(src_mask, diagonal=mask).logical_not()

        query2 = self.masked_attn_head(
            query,
            key,
            values,
            mask=src_mask,
            zero_pad=(mask == 0),
        )
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.cic(query)
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """双流架构中的单个 Transformer 堆叠"""

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        k1: int,
        k2: int,
        seq_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                    kq_same=kq_same,
                    k1=k1,
                    k2=k2,
                    max_seq_len=seq_len,
                )
                for _ in range(n_blocks)
            ]
        )
        self.position_emb = CosinePositionalEmbedding(
            d_model=self.d_model, max_len=seq_len
        )

    def forward(
        self, q_embed_data: torch.Tensor, qa_embed_data: torch.Tensor
    ) -> torch.Tensor:
        # 添加位置编码
        q_embed_data = q_embed_data + self.position_emb(q_embed_data)
        qa_embed_data = qa_embed_data + self.position_emb(qa_embed_data)

        y = qa_embed_data  # values: 交互表示
        x = q_embed_data  # query/key: 内容表示

        for block in self.blocks:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


class MTKT(nn.Module):
    """MTKT模型

    双流架构:
    - 内容流 (model): 处理技能/交互嵌入
    - 时间流 (model2): 处理时间间隔嵌入
    - 门控融合: w = sigmoid(c_weight + t_weight), output = w*content + (1-w)*temporal

    预测语义:
    - preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]
    - 同位置输出（trainer 用 same_position=True 适配 next-item 对齐）
    """

    def __init__(
        self,
        num_skills: int,
        n_pid: int = 0,
        num_rgap: int = 100,
        num_sgap: int = 100,
        num_pcount: int = 15,
        d_model: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.2,
        d_ff: int = 256,
        kq_same: int = 1,
        separate_qa: bool = False,
        l2: float = 1e-5,
        k1: int = 1,
        k2: int = 3,
        num_attn_heads: int = 8,
        final_fc_dim: int = 512,
        final_fc_dim2: int = 256,
        seq_len: int = 200,
    ):
        super().__init__()
        self.model_name = "mtkt"
        self.num_skills = num_skills
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = separate_qa

        embed_l = d_model

        # --- 技能和交互嵌入 ---
        self.q_embed = nn.Embedding(num_skills, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * num_skills + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        # --- Rasch 难度嵌入 (题目级) ---
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_skills + 1, embed_l)

        # --- 时间间隔嵌入 ---
        self.time_emb = TimeGapEmbedding(num_rgap, num_sgap, num_pcount, d_model)

        # --- 双流 Transformer ---
        self.model = Architecture(  # 内容流
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=(kq_same == 1),
            k1=k1,
            k2=k2,
            seq_len=seq_len,
        )
        self.model2 = Architecture(  # 时间流
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=(kq_same == 1),
            k1=k1,
            k2=k2,
            seq_len=seq_len,
        )

        # --- 门控融合 ---
        self.c_weight = nn.Linear(d_model, d_model)
        self.t_weight = nn.Linear(d_model, d_model)

        # --- 输出头 ---
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
        """Initialize Rasch difficulty parameters to zero."""
        if self.n_pid > 0:
            torch.nn.init.constant_(self.difficult_param.weight, 0.0)

    def base_emb(
        self,
        q_data: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算基础嵌入

        Returns:
            q_embed_data: 技能嵌入 [B, S, d_model]
            qa_embed_data: 交互嵌入 [B, S, d_model]
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
        question: torch.Tensor,
        rgap: torch.Tensor,
        sgap: torch.Tensor,
        pcount: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            sequence: 技能 ID 序列 [B, S]
            response: 响应序列 [B, S]
            question: 题目 ID 序列 [B, S]
            rgap: 复习间隔 [B, S]
            sgap: 连续间隔 [B, S]
            pcount: 练习次数 [B, S]

        Returns:
            preds: 预测概率 [B, S]
            c_reg_loss: Rasch 正则化损失
        """
        q_data = sequence  # 技能 ID
        target = response  # 响应标签
        pid_data = question  # 题目 ID

        # 1. 基础嵌入
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        # 2. Rasch 难度调制
        c_reg_loss = torch.tensor(0.0, device=sequence.device)
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            c_reg_loss = (pid_embed_data**2).sum() * self.l2

        # 3. 时间间隔嵌入
        temb = self.time_emb(rgap, sgap, pcount)

        # 4. 内容流 Transformer
        d_output = self.model(q_embed_data, qa_embed_data)

        # 5. 时间流 Transformer
        t_out = self.model2(temb, qa_embed_data)

        # 6. 门控融合
        w = torch.sigmoid(self.c_weight(d_output) + self.t_weight(t_out))
        d_output = w * d_output + (1 - w) * t_out

        # 7. 加入时间嵌入
        q_embed_data = q_embed_data + temb

        # 8. 输出
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        return preds, c_reg_loss
