"""FlucKT 模型实现：AKT 双塔架构 + 因果卷积频率滤波 + Kerple log kernel 位置偏置"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter


class CausalConv1d(nn.Module):
    """因果一维卷积，切掉右侧填充保证不泄露未来信息"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

    def forward(self, x):
        return self.conv(x)[:, :, : -self.conv.padding[0]]


class FrequencyLayer(nn.Module):
    """频率分解层：因果卷积作低通滤波，可学习 sqrt_beta^2 控制高通分量权重"""

    def __init__(self, dropout, hidden_size, kernel_size):
        super().__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalConv1d(hidden_size, hidden_size, kernel_size)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # input_tensor: [B, S, H]
        low_pass = self.causal_conv(input_tensor.transpose(1, 2)).transpose(1, 2)
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + self.sqrt_beta**2 * high_pass
        hidden_states = self.out_dropout(sequence_emb_fft)
        return self.layer_norm(hidden_states + input_tensor)


class ParallelKerpleLog(nn.Module):
    """Kerple log kernel 位置偏置：-p * log(1 + a * |i - j|)，p/a 由 clamp 保持正定"""

    def __init__(self, num_attention_heads):
        super().__init__()
        self.eps = 1e-2
        self.bias_p = Parameter(torch.rand(num_attention_heads, 1, 1) * 2)
        self.bias_a = Parameter(torch.rand(num_attention_heads, 1, 1))
        self.register_buffer("_cached_diff", None, persistent=False)
        self._cached_seq_len = 0

    def _build_diff(self, seq_len, device, dtype):
        # tril of (i - j): query i attends key j with distance i - j
        diff = torch.tril(
            torch.arange(seq_len, device=device).view(seq_len, 1).repeat(1, seq_len)
            + torch.arange(0, -seq_len, -1, device=device)
        )
        self._cached_diff = diff.to(dtype)
        self._cached_seq_len = seq_len

    def forward(self, x):
        seq_len = x.shape[-1]
        diff = self._cached_diff
        if (
            self._cached_seq_len != seq_len
            or diff is None
            or diff.device != x.device
            or diff.dtype != x.dtype
        ):
            self._build_diff(seq_len, x.device, x.dtype)
            diff = self._cached_diff
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        return x - self.bias_p * torch.log(1 + self.bias_a * diff)


def attention(
    q,
    k,
    v,
    d_k,
    mask,
    dropout,
    zero_pad,
    gamma=None,
    kernel_bias=None,
    position_effect=None,
    use_exp_decay=True,
):
    """注意力计算

    Args:
        q: Query张量 [BS, n_heads, seq_len, d_k]
        k: Key张量 [BS, n_heads, seq_len, d_k]
        v: Value张量 [BS, n_heads, seq_len, d_k]
        d_k: 每个头的维度
        mask: 掩码矩阵
        dropout: Dropout层
        zero_pad: 是否对第一行进行零填充
        gamma: 位置衰减参数
        kernel_bias: ParallelKerpleLog 模块或None
        position_effect: 位置差值矩阵
        use_exp_decay: 是否启用单调指数衰减

    Returns:
        output: 注意力输出 [BS, seq_len, d_model]
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if use_exp_decay:
        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)
            scores_.masked_fill_(mask == 0, 0.0)
            dist_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            del scores_
            torch.sub(disttotal_scores, dist_scores, out=dist_scores)
            del disttotal_scores
            dist_scores.mul_(position_effect).clamp_min_(0.0).sqrt_()

        gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
        del dist_scores
        scores = scores * total_effect
        del total_effect

    if kernel_bias is not None:
        scores = kernel_bias(scores)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    """多头注意力机制，支持指数衰减与 Kerple kernel 位置偏置"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        use_exp_decay: bool = True,
        use_kernel_bias: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.use_exp_decay = use_exp_decay

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

        self.kernel_bias = ParallelKerpleLog(n_heads) if use_kernel_bias else None

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

    def forward(self, q, k, v, mask, zero_pad, position_effect=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(
            q,
            k,
            v,
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            gamma=self.gammas,
            kernel_bias=self.kernel_bias,
            position_effect=position_effect,
            use_exp_decay=self.use_exp_decay,
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    """Transformer层，包含多头注意力和前馈网络"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        use_exp_decay: bool = True,
        use_kernel_bias: bool = False,
    ):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=kq_same,
            use_exp_decay=use_exp_decay,
            use_kernel_bias=use_kernel_bias,
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Seqlen-dependent constant tensors owned by the layer. Registered as
        # non-persistent buffers so they migrate with .to(device) yet stay out
        # of state_dict (rebuilt on first forward).
        self.register_buffer("_causal_mask_k0", None, persistent=False)
        self.register_buffer("_causal_mask_k1", None, persistent=False)
        self.register_buffer("_position_effect", None, persistent=False)
        self._const_seqlen = 0

    def _build_constants(self, seqlen: int, device: torch.device) -> None:
        ones = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device)
        self._causal_mask_k1 = ones.tril().view(1, 1, seqlen, seqlen)
        self._causal_mask_k0 = ones.tril(diagonal=-1).view(1, 1, seqlen, seqlen)
        idx = torch.arange(seqlen, device=device, dtype=torch.float32)
        self._position_effect = torch.abs(
            idx.view(seqlen, 1) - idx.view(1, seqlen)
        ).view(1, 1, seqlen, seqlen)
        self._const_seqlen = seqlen

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen = query.size(1)
        pe = self._position_effect
        if pe is None or seqlen != self._const_seqlen or pe.device != query.device:
            self._build_constants(seqlen, query.device)
            pe = self._position_effect
        src_mask = self._causal_mask_k1 if mask else self._causal_mask_k0

        query2 = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=mask == 0, position_effect=pe
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """FlucKT架构：可选频率滤波层 + 双流Transformer（QA编码器和知识检索器）"""

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        use_frequency_layer: bool,
        use_kernel_bias: bool,
        use_exp_decay: bool,
        kernel_size: int,
    ):
        super().__init__()
        self.use_frequency_layer = use_frequency_layer

        if use_frequency_layer:
            self.filter_layer = FrequencyLayer(dropout, d_model, kernel_size)

        layer_kwargs = {
            "d_model": d_model,
            "d_feature": d_model // n_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "n_heads": n_heads,
            "kq_same": kq_same,
            "use_exp_decay": use_exp_decay,
            "use_kernel_bias": use_kernel_bias,
        }
        self.blocks_1 = nn.ModuleList(
            [TransformerLayer(**layer_kwargs) for _ in range(n_blocks)]
        )
        self.blocks_2 = nn.ModuleList(
            [TransformerLayer(**layer_kwargs) for _ in range(n_blocks * 2)]
        )

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data=None):
        y = qa_embed_data
        x = q_embed_data

        if self.use_frequency_layer:
            x = self.filter_layer(x)
            y = self.filter_layer(y)

        # Encoder: 对0~t-1时刻前的qa信息进行编码
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y)

        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # Peek current question
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # Don't peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class FlucKT(nn.Module):
    """FlucKT 模型

    基于AKT双流Transformer架构，叠加频率感知组件：
    - FrequencyLayer: 因果卷积低通滤波 + 可学习高通权重
    - ParallelKerpleLog: Kerple log kernel 位置偏置
    - 指数衰减注意力（可开关）

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
        use_frequency_layer: 是否启用频率分解层
        use_kernel_bias: 是否启用Kerple log kernel位置偏置
        use_exp_decay: 是否启用单调指数衰减注意力
        kernel_size: 因果卷积核大小
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        d_model: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.05,
        d_ff: int = 256,
        kq_same: int = 1,
        final_fc_dim: int = 512,
        num_attn_heads: int = 4,
        separate_qa: bool = False,
        l2: float = 1e-5,
        use_frequency_layer: bool = True,
        use_kernel_bias: bool = True,
        use_exp_decay: bool = True,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.model_name = "fluckt"
        self.num_c = num_c
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = separate_qa
        embed_l = d_model

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
        self.q_embed = nn.Embedding(self.num_c, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.num_c + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=bool(kq_same),
            use_frequency_layer=use_frequency_layer,
            use_kernel_bias=use_kernel_bias,
            use_exp_decay=use_exp_decay,
            kernel_size=kernel_size,
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

    def reset(self):
        """初始化Rasch模型参数为0"""
        if self.n_pid > 0:
            for p in self.parameters():
                if p.size(0) == self.n_pid + 1:
                    torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        """计算基础嵌入

        Args:
            q_data: 技能ID序列
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
        q_embed_data, qa_embed_data = self.base_emb(sequence, response)

        pid_embed_data = None
        c_reg_loss = torch.tensor(0.0, device=sequence.device)

        if self.n_pid > 0 and pid_data is not None:
            q_embed_diff_data = self.q_embed_diff(sequence)  # d_ct
            pid_embed_data = self.difficult_param(pid_data)  # uq (题目难度)

            # question encoder: uq * d_ct + c_ct
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(response)  # h_rt

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

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        return preds, c_reg_loss
