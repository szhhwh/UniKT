"""AKT (Attentive Knowledge Tracing) 模型实现

原始论文: Ghosh et al., "Context-Aware Attentive Knowledge Tracing", KDD 2020
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def attention(
    q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None, position_effect=None
):
    """
    注意力计算函数

    Args:
        q: Query张量 [BS, n_heads, seq_len, d_k]
        k: Key张量 [BS, n_heads, seq_len, d_k]
        v: Value张量 [BS, n_heads, seq_len, d_k]
        d_k: 每个头的维度
        mask: 掩码矩阵
        dropout: Dropout层
        zero_pad: 是否对第一行进行零填充
        gamma: 位置衰减参数
        pdiff: 题目难度差异（暂不使用）

    Returns:
        output: 注意力输出 [BS, n_heads, seq_len, d_k]
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    masked_positions = mask == 0

    with torch.no_grad():
        scores_ = scores.masked_fill(masked_positions, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_.masked_fill_(masked_positions, 0.0)
        dist_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        del scores_
        torch.sub(disttotal_scores, dist_scores, out=dist_scores)
        del disttotal_scores
        dist_scores.mul_(position_effect).clamp_min_(0.0).sqrt_()

    gamma = -1.0 * F.softplus(gamma).unsqueeze(0)

    if pdiff is None:
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(
            pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2]
        )
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(
            (dist_scores * gamma * diff).exp(), min=1e-5, max=1e5
        )
    del dist_scores
    scores = scores * total_effect
    del total_effect

    scores.masked_fill_(masked_positions, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        bias: bool = True,
        emb_type: str = "qid",
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
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
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            xavier_uniform_(self.gammas)
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

    def forward(self, q, k, v, mask, zero_pad, pdiff=None, position_effect=None):
        bs = q.size(0)

        if self.emb_type.startswith("qid"):
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention(
                q,
                k,
                v,
                self.d_k,
                mask,
                self.dropout,
                zero_pad,
                gammas,
                pdiff,
                position_effect=position_effect,
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
        kq_same: bool,
        emb_type: str,
    ):
        super().__init__()
        kq_same = kq_same == 1

        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type
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

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        seqlen = query.size(1)
        pe = self._position_effect
        if pe is None or seqlen != self._const_seqlen or pe.device != query.device:
            self._build_constants(seqlen, query.device)
            pe = self._position_effect
        src_mask = self._causal_mask_k1 if mask else self._causal_mask_k0

        query2 = self.masked_attn_head(
            query,
            key,
            values,
            mask=src_mask,
            zero_pad=mask == 0,
            pdiff=pdiff,
            position_effect=pe,
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """AKT架构，包含QA编码器和知识检索器"""

    def __init__(
        self,
        n_question: int,
        n_blocks: int,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        model_type: str,
        emb_type: str,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {"akt"}:
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

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data=None):
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        # Encoder: 对0~t-1时刻前的qa信息进行编码
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)

        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # Peek current question
                x = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    apply_pos=False,
                    pdiff=pid_embed_data,
                )
                flag_first = False
            else:  # Don't peek current response
                x = block(
                    mask=0,
                    query=x,
                    key=x,
                    values=y,
                    apply_pos=True,
                    pdiff=pid_embed_data,
                )
                flag_first = True
        return x


class AKT(nn.Module):
    """AKT (Attentive Knowledge Tracing) 模型

    基于上下文感知的知识追踪模型，使用双流Transformer架构：
    - blocks_1: QA编码器，编码历史交互信息
    - blocks_2: 知识检索器，从历史中检索相关知识状态

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
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        d_model: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.2,
        d_ff: int = 256,
        kq_same: int = 1,
        final_fc_dim: int = 512,
        num_attn_heads: int = 8,
        separate_qa: bool = False,
        l2: float = 1e-5,
        emb_type: str = "qid",
    ):
        super().__init__()
        self.model_name = "akt"
        self.num_c = num_c
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
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

        # Architecture: 双流Transformer
        self.model = Architecture(
            n_question=self.num_c,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
            emb_type=self.emb_type,
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

    def base_emb(self, q_data, target, pid_data=None):
        """计算基础嵌入

        Args:
            q_data: 问题ID（技能ID）序列
            target: 响应序列
            pid_data: Problem ID序列（可选）

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
        q_embed_data, qa_embed_data = self.base_emb(sequence, response, pid_data)

        # 处理Problem ID和Rasch模型
        pid_embed_data = None
        c_reg_loss = torch.tensor(0.0, device=sequence.device)

        if self.n_pid > 0 and pid_data is not None:
            # 计算问题差异向量嵌入
            q_embed_diff_data = self.q_embed_diff(sequence)  # d_ct
            pid_embed_data = self.difficult_param(pid_data)  # uq (题目难度)

            # question encoder: uq * d_ct + c_ct
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            # 计算交互差异向量嵌入
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

        # 通过双流Transformer架构
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        # 拼接输出和问题嵌入
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        # Sigmoid激活
        preds = torch.sigmoid(output)

        # AKT输出语义说明：
        # AKT的第二个流（blocks_2）第一层可以"peek current question"
        # 因此 preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]

        return preds, c_reg_loss
