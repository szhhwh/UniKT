"""UKT (Uncertainty-aware Knowledge Tracing) 模型实现

使用双流不确定性表示（均值+协方差）和 Wasserstein 距离注意力机制的知识追踪模型。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def wasserstein_distance(mean1, cov1, mean2, cov2):
    """计算两个分布之间的 Wasserstein 距离"""
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)
    return ret


def d2s_1overx(distance):
    """距离到相似度转换"""
    return 1 / (1 + distance)


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    """批量 Wasserstein 距离矩阵计算"""
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = (
        -2 * torch.matmul(mean1, mean2.transpose(-1, -2))
        + mean1_2
        + mean2_2.transpose(-1, -2)
    )

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = (
        -2
        * torch.matmul(
            torch.sqrt(torch.clamp(cov1, min=1e-24)),
            torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2),
        )
        + cov1_2
        + cov2_2.transpose(-1, -2)
    )

    return ret + cov_ret


def _compute_position_effect(max_len):
    """预计算位置效果矩阵 |i - j|，作为 buffer 缓存

    Args:
        max_len: 最大序列长度

    Returns:
        position_effect: [max_len, max_len] 的 FloatTensor
    """
    idx = torch.arange(max_len)
    return torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)).float()


def _compute_dist_scores(scores, mask, position_effect):
    """计算距离衰减分数

    dp_attention 和 w2_attention 共享的距离效果计算。

    Args:
        scores: 注意力分数 [B, H, S, S]
        mask: 因果掩码 [1, 1, S, S] 或 [B, H, S, S]
        position_effect: 预计算的位置效果矩阵 [S, S]

    Returns:
        dist_scores: 距离衰减分数 [B, H, S, S]（已 detach）
    """
    with torch.no_grad():
        probabilities = F.softmax(scores.masked_fill(mask == 0, -1e32), dim=-1)
        probabilities.masked_fill_(mask == 0, 0.0)
        distance = torch.cumsum(probabilities, dim=-1)
        total = probabilities.sum(dim=-1, keepdim=True)
        del probabilities
        torch.sub(total, distance, out=distance)
        del total
        length = scores.size(-1)
        position = position_effect[:length, :length].to(scores.device)
        return distance.mul_(position).clamp_min_(0.0).sqrt_()


def _apply_position_effect(scores, dist_scores, gamma, softplus):
    """应用位置效果到注意力分数

    Args:
        scores: 原始注意力分数
        dist_scores: 距离衰减分数
        gamma: 可学习的 gamma 参数 [n_heads, 1, 1]
        softplus: Softplus 模块

    Returns:
        修改后的注意力分数
    """
    gamma_val = -1.0 * softplus(gamma).unsqueeze(0)
    total_effect = torch.clamp((dist_scores * gamma_val).exp(), min=1e-5, max=1e5)
    return scores * total_effect


def dp_attention(
    q_mean,
    q_cov,
    k_mean,
    k_cov,
    v_mean,
    v_cov,
    d_k,
    mask,
    dropout,
    zero_pad,
    gamma,
    position_effect,
    softplus,
):
    """双流点积注意力"""
    scores_mean = torch.matmul(q_mean, k_mean.transpose(-2, -1)) / math.sqrt(d_k)
    scores_cov = torch.matmul(q_cov, k_cov.transpose(-2, -1)) / math.sqrt(d_k)

    # 双流分别计算距离衰减
    dist_scores_mean = _compute_dist_scores(scores_mean, mask, position_effect)
    dist_scores_cov = _compute_dist_scores(scores_cov, mask, position_effect)

    scores_mean = _apply_position_effect(scores_mean, dist_scores_mean, gamma, softplus)
    scores_cov = _apply_position_effect(scores_cov, dist_scores_cov, gamma, softplus)

    scores_mean.masked_fill_(mask == 0, -1e32)
    scores_cov.masked_fill_(mask == 0, -1e32)
    scores_mean = F.softmax(scores_mean, dim=-1)
    scores_cov = F.softmax(scores_cov, dim=-1)

    scores_mean = dropout(scores_mean)
    scores_cov = dropout(scores_cov)

    output_mean = torch.matmul(scores_mean, v_mean)
    output_cov = torch.matmul(scores_cov, v_cov)
    if zero_pad:
        output_mean[:, :, 0, :].zero_()
        output_cov[:, :, 0, :].zero_()
    return output_mean, output_cov


def w2_attention(
    q_mean,
    q_cov,
    k_mean,
    k_cov,
    v_mean,
    v_cov,
    d_k,
    mask,
    dropout,
    zero_pad,
    gamma,
    position_effect,
    softplus,
):
    """Wasserstein 距离注意力"""
    scores = -wasserstein_distance_matmul(q_mean, q_cov, k_mean, k_cov) / math.sqrt(d_k)

    dist_scores = _compute_dist_scores(scores, mask, position_effect)
    scores = _apply_position_effect(scores, dist_scores, gamma, softplus)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    scores = dropout(scores)

    output_mean = torch.matmul(scores, v_mean)
    output_cov = torch.matmul(scores**2, v_cov)
    if zero_pad:
        output_mean[:, :, 0, :].zero_()
        output_cov[:, :, 0, :].zero_()
    return output_mean, output_cov


class WassersteinNCELoss(nn.Module):
    """Wasserstein 距离对比学习损失"""

    def __init__(self, temperature):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.activation = nn.ELU()

    def forward(self, batch_one_mean, batch_one_cov, batch_two_mean, batch_two_cov):
        batch_one_cov = self.activation(batch_one_cov) + 1
        batch_two_cov = self.activation(batch_two_cov) + 1

        sim11 = (
            d2s_1overx(
                wasserstein_distance_matmul(
                    batch_one_mean, batch_one_cov, batch_one_mean, batch_one_cov
                )
            )
            / self.temperature
        )
        sim22 = (
            d2s_1overx(
                wasserstein_distance_matmul(
                    batch_two_mean, batch_two_cov, batch_two_mean, batch_two_cov
                )
            )
            / self.temperature
        )
        sim12 = (
            -d2s_1overx(
                wasserstein_distance_matmul(
                    batch_one_mean, batch_one_cov, batch_two_mean, batch_two_cov
                )
            )
            / self.temperature
        )

        d = sim12.shape[-1]
        device = sim12.device
        sim11[..., range(d), range(d)] = float("-inf")
        sim22[..., range(d), range(d)] = float("-inf")
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=device)
        return self.criterion(logits, labels)


class CosinePositionalEmbedding(nn.Module):
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


class MultiHeadAttention(nn.Module):
    """双流多头注意力机制"""

    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        if not kq_same:
            self.q_mean_linear = nn.Linear(d_model, d_model, bias=bias)
            self.q_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_mean_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_cov_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_mean_linear.weight)
        xavier_uniform_(self.k_cov_linear.weight)
        xavier_uniform_(self.v_mean_linear.weight)
        xavier_uniform_(self.v_cov_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_mean_linear.weight)
            xavier_uniform_(self.q_cov_linear.weight)
        if self.proj_bias:
            constant_(self.k_mean_linear.bias, 0.0)
            constant_(self.k_cov_linear.bias, 0.0)
            constant_(self.v_mean_linear.bias, 0.0)
            constant_(self.v_cov_linear.bias, 0.0)
            if not self.kq_same:
                constant_(self.q_mean_linear.bias, 0.0)
                constant_(self.q_cov_linear.bias, 0.0)
            constant_(self.out_mean_proj.bias, 0.0)
            constant_(self.out_cov_proj.bias, 0.0)

    def forward(
        self,
        q_mean,
        q_cov,
        k_mean,
        k_cov,
        v_mean,
        v_cov,
        mask,
        atten_type,
        zero_pad,
        position_effect,
        softplus,
    ):
        bs = q_mean.size(0)

        k_mean = self.k_mean_linear(k_mean).view(bs, -1, self.h, self.d_k)
        k_cov = self.k_cov_linear(k_cov).view(bs, -1, self.h, self.d_k)

        if self.kq_same:
            q_mean = self.k_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.k_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)
        else:
            q_mean = self.q_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.q_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)

        v_mean = self.v_mean_linear(v_mean).view(bs, -1, self.h, self.d_k)
        v_cov = self.v_cov_linear(v_cov).view(bs, -1, self.h, self.d_k)

        k_mean = k_mean.transpose(1, 2)
        q_mean = q_mean.transpose(1, 2)
        v_mean = v_mean.transpose(1, 2)
        k_cov = k_cov.transpose(1, 2)
        q_cov = q_cov.transpose(1, 2)
        v_cov = v_cov.transpose(1, 2)

        attn_fn = w2_attention if atten_type == "w2" else dp_attention
        scores_mean, scores_cov = attn_fn(
            q_mean,
            q_cov,
            k_mean,
            k_cov,
            v_mean,
            v_cov,
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            self.gammas,
            position_effect,
            softplus,
        )

        concat_mean = (
            scores_mean.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        )
        concat_cov = scores_cov.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output_mean = self.out_mean_proj(concat_mean)
        output_cov = self.out_cov_proj(concat_cov)
        return output_mean, output_cov


class TransformerLayer(nn.Module):
    """双流 Transformer 层"""

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mean_linear1 = nn.Linear(d_model, d_ff)
        self.cov_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mean_linear2 = nn.Linear(d_ff, d_model)
        self.cov_linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.ELU()

    def forward(
        self,
        mask,
        query_mean,
        query_cov,
        key_mean,
        key_cov,
        values_mean,
        values_cov,
        atten_type="w2",
        apply_pos=True,
        position_effect=None,
        softplus=None,
    ):
        device = query_mean.device
        seqlen = query_mean.size(1)

        src_mask = (
            torch.tril(
                torch.ones(seqlen, seqlen, device=device, dtype=torch.uint8),
                diagonal=mask - 1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .bool()
        )

        zero_pad = mask == 0
        query2_mean, query2_cov = self.masked_attn_head(
            query_mean,
            query_cov,
            key_mean,
            key_cov,
            values_mean,
            values_cov,
            mask=src_mask,
            atten_type=atten_type,
            zero_pad=zero_pad,
            position_effect=position_effect,
            softplus=softplus,
        )

        query_mean = query_mean + self.dropout1(query2_mean)
        query_cov = query_cov + self.dropout1(query2_cov)
        query_mean = self.layer_norm1(query_mean)
        query_cov = self.layer_norm1(self.activation2(query_cov) + 1)

        if apply_pos:
            query2_mean = self.mean_linear2(
                self.dropout(self.activation(self.mean_linear1(query_mean)))
            )
            query2_cov = self.cov_linear2(
                self.dropout(self.activation(self.cov_linear1(query_cov)))
            )
            query_mean = query_mean + self.dropout2(query2_mean)
            query_cov = query_cov + self.dropout2(query2_cov)
            query_mean = self.layer_norm2(query2_mean)
            query_cov = self.layer_norm2(self.activation2(query2_cov) + 1)

        return query_mean, query_cov


class Architecture(nn.Module):
    """UKT 架构：双流 Transformer 编码器"""

    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, kq_same, seq_len):
        super().__init__()
        self.d_model = d_model

        self.position_mean_embeddings = CosinePositionalEmbedding(
            d_model=self.d_model, max_len=seq_len
        )
        self.position_cov_embeddings = CosinePositionalEmbedding(
            d_model=self.d_model, max_len=seq_len
        )

        self.blocks = nn.ModuleList(
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

        self.register_buffer(
            "_position_effect", _compute_position_effect(seq_len), persistent=False
        )
        self._softplus = nn.Softplus()

    def forward(
        self, q_mean_embed, q_cov_embed, qa_mean_embed, qa_cov_embed, atten_type="w2"
    ):
        # 添加位置编码
        q_mean_embed = q_mean_embed + self.position_mean_embeddings(q_mean_embed)
        q_cov_embed = q_cov_embed + self.position_cov_embeddings(q_cov_embed)
        qa_mean_embed = qa_mean_embed + self.position_mean_embeddings(qa_mean_embed)
        qa_cov_embed = qa_cov_embed + self.position_cov_embeddings(qa_cov_embed)

        # ELU 激活协方差流
        q_cov_embed = F.elu(q_cov_embed) + 1
        qa_cov_embed = F.elu(qa_cov_embed) + 1

        # 交互嵌入作为 value，技能嵌入作为 query/key
        y_mean, y_cov = qa_mean_embed, qa_cov_embed
        x_mean, x_cov = q_mean_embed, q_cov_embed

        for block in self.blocks:
            x_mean, x_cov = block(
                mask=0,
                query_mean=x_mean,
                query_cov=x_cov,
                key_mean=x_mean,
                key_cov=x_cov,
                values_mean=y_mean,
                values_cov=y_cov,
                atten_type=atten_type,
                apply_pos=True,
                position_effect=self._position_effect,
                softplus=self._softplus,
            )

        return x_mean, x_cov


class UKT(nn.Module):
    """UKT (Uncertainty-aware Knowledge Tracing) 模型

    使用双流不确定性表示（均值+协方差）和 Wasserstein 注意力机制。

    Args:
        num_c: 概念（技能）数量
        n_pid: Problem ID数量，0表示不使用
        d_model: 模型隐藏维度
        n_blocks: Transformer块数量
        dropout: Dropout概率
        d_ff: 前馈网络维度
        final_fc_dim: 第一层全连接维度
        final_fc_dim2: 第二层全连接维度
        num_attn_heads: 注意力头数量
        kq_same: Key和Query是否使用相同的线性变换
        separate_qa: 是否使用独立的QA嵌入
        use_CL: 是否启用对比学习
        cl_weight: 对比学习损失权重
        use_uncertainty_aug: 是否使用不确定性增强
        l2: L2正则化系数
        emb_type: 嵌入类型
        atten_type: 注意力类型
        seq_len: 最大序列长度
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        d_model: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.2,
        d_ff: int = 512,
        final_fc_dim: int = 512,
        final_fc_dim2: int = 256,
        num_attn_heads: int = 8,
        kq_same: int = 1,
        separate_qa: bool = False,
        use_CL: bool = True,
        cl_weight: float = 0.02,
        use_uncertainty_aug: bool = True,
        l2: float = 1e-5,
        emb_type: str = "stoc_qid",
        atten_type: str = "w2",
        seq_len: int = 200,
    ):
        super().__init__()
        self.model_name = "ukt"
        self.num_c = num_c
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.use_CL = use_CL
        self.use_uncertainty_aug = use_uncertainty_aug
        self.atten_type = atten_type

        embed_l = d_model

        if use_CL:
            self.wloss = WassersteinNCELoss(1)
            self.cl_weight = cl_weight

        # 概念嵌入（均值+协方差双流）
        self.mean_q_embed = nn.Embedding(num_c, embed_l)
        self.cov_q_embed = nn.Embedding(num_c, embed_l)

        if separate_qa:
            self.mean_qa_embed = nn.Embedding(2 * num_c + 1, embed_l)
            self.cov_qa_embed = nn.Embedding(2 * num_c + 1, embed_l)
        else:
            self.mean_qa_embed = nn.Embedding(2, embed_l)
            self.cov_qa_embed = nn.Embedding(2, embed_l)

        # Problem ID 相关嵌入（Rasch 模型）
        if n_pid > 0:
            if "scalar" in emb_type:
                self.difficult_param = nn.Embedding(n_pid + 1, 1)
            else:
                self.difficult_param = nn.Embedding(n_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(num_c + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * num_c + 1, embed_l)

        # Transformer 架构
        self.model = Architecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=self.kq_same,
            seq_len=seq_len,
        )

        # 输出层
        self.out = nn.Sequential(
            nn.Linear(embed_l * 4, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        self.reset()

    def reset(self):
        """初始化 Rasch 模型参数"""
        if self.n_pid > 0:
            for p in self.parameters():
                if p.size(0) == self.n_pid + 1:
                    torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        """计算双流基础嵌入

        Args:
            q_data: 概念ID序列 [B, L]
            target: 响应序列 [B, L]

        Returns:
            (q_mean_embed, q_cov_embed, qa_mean_embed, qa_cov_embed)
        """
        q_mean_embed = self.mean_q_embed(q_data)
        q_cov_embed = self.cov_q_embed(q_data)

        if self.separate_qa:
            qa_data = q_data + self.num_c * target
            qa_mean_embed = self.mean_qa_embed(qa_data)
            qa_cov_embed = self.cov_qa_embed(qa_data)
        else:
            qa_mean_embed = self.mean_qa_embed(target) + q_mean_embed
            qa_cov_embed = self.cov_qa_embed(target) + q_cov_embed

        return q_mean_embed, q_cov_embed, qa_mean_embed, qa_cov_embed

    @staticmethod
    def _augment_response(response, mask):
        """创建对比学习用的增强响应序列

        增强算法:
        - r_aug[0] 在 rshft_aug_mask[-1] == r_aug_mask[0] 时翻转
        - rshft_aug[0] 在 rshft_aug[0] == rshft_aug_mask[-1] 时翻转

        Note: 原版循环存在自引用 bug (rshft_aug[0] 在 i=0 时被修改后,
        后续迭代条件失效)，所以实际上只有 rshft_aug[0] (= position 1 of target_aug) 被翻转。

        Args:
            response: 响应序列 [B, L]
            mask: 有效性掩码 [B, L]

        Returns:
            增强后的响应序列 [B, L]
        """
        B, L = response.shape
        device = response.device

        aug = response.clone()
        vmask = mask.bool()

        # 相邻有效性掩码 [B, L-1] (匹配原版 mseqs)
        adj_mask = vmask[:, :-1] & vmask[:, 1:]

        # 每个样本的有效相邻对数量
        n_valid = adj_mask.sum(dim=1)  # [B]
        has_pairs = n_valid > 0

        if not has_pairs.any():
            return aug

        # 找到每个 batch 的 first/last adj_mask=True 索引
        arange = torch.arange(adj_mask.size(1), device=device, dtype=torch.float)
        arange_expanded = arange.unsqueeze(0).expand(B, -1)
        inf_t = torch.tensor(float("inf"), device=device)
        neg_inf_t = torch.tensor(float("-inf"), device=device)

        first_idx = torch.where(adj_mask, arange_expanded, inf_t).argmin(dim=1)
        last_idx = torch.where(adj_mask, arange_expanded, neg_inf_t).argmax(dim=1)

        batch_idx = torch.arange(B, device=device)

        last_valid_shft = response[batch_idx, (last_idx + 1).clamp(max=L - 1)].float()
        first_valid_r = response[batch_idx, first_idx].float()

        # rshft_aug[0] = response[1] * mseqs[0] (first pair invalid 时置 0)
        rshft_aug_0 = response[:, 1].float() * adj_mask[:, 0].float()

        # 条件 1: 翻转位置 0 (r_aug[0])
        flip_pos0 = (last_valid_shft == first_valid_r) & has_pairs
        aug[:, 0] = torch.where(flip_pos0, 1 - aug[:, 0], aug[:, 0])

        # 条件 2: 翻转位置 1 (rshft_aug[0])
        # 原版条件: rshft_aug[0] == rshft_aug_mask[-1]
        flip_pos1 = (rshft_aug_0 == last_valid_shft) & has_pairs
        aug[:, 1] = torch.where(flip_pos1, 1 - aug[:, 1], aug[:, 1])

        return aug

    def forward(self, sequence, response, mask=None, pid_data=None, target_aug=None):
        """前向传播

        Args:
            sequence: 技能ID序列 [B, L]
            response: 响应序列 [B, L]
            mask: 有效性掩码 [B, L]
            pid_data: 题目ID序列 [B, L]，0保留给padding
            target_aug: 对比学习增强响应序列 [B, L]

        Returns:
            preds: 预测概率 [B, L]
            cl_loss: 对比学习损失（标量）
            cov_stats: 协方差统计信息 [B]
        """
        q_data = sequence
        target = response

        # 双流嵌入
        q_mean_embed, q_cov_embed, qa_mean_embed, qa_cov_embed = self.base_emb(
            q_data, target
        )

        # 对比学习：计算增强嵌入
        cl_loss = torch.tensor(0.0, device=sequence.device)
        do_cl = self.training and self.use_CL

        if do_cl:
            if target_aug is not None:
                target_aug = target_aug.to(device=response.device, dtype=response.dtype)
            elif self.use_uncertainty_aug and mask is not None:
                target_aug = self._augment_response(response, mask)
            else:
                target_aug = target

            (
                aug_q_mean_embed,
                aug_q_cov_embed,
                aug_qa_mean_embed,
                aug_qa_cov_embed,
            ) = self.base_emb(q_data, target_aug)

        # Problem ID (Rasch 模型)
        c_reg_loss = torch.tensor(0.0, device=sequence.device)
        if self.n_pid > 0 and pid_data is not None and "norasch" not in self.emb_type:
            q_embed_diff = self.q_embed_diff(q_data)
            pid_embed = self.difficult_param(pid_data)
            q_mean_embed = q_mean_embed + pid_embed * q_embed_diff
            q_cov_embed = q_cov_embed + pid_embed * q_embed_diff

            if do_cl:
                aug_q_mean_embed = aug_q_mean_embed + pid_embed * q_embed_diff
                aug_q_cov_embed = aug_q_cov_embed + pid_embed * q_embed_diff

            if "aktrasch" in self.emb_type:
                qa_embed_diff = self.qa_embed_diff(target)
                qa_mean_embed = qa_mean_embed + pid_embed * (
                    qa_embed_diff + q_embed_diff
                )
                qa_cov_embed = qa_cov_embed + pid_embed * (qa_embed_diff + q_embed_diff)
                if do_cl:
                    aug_qa_embed_diff = self.qa_embed_diff(target_aug)
                    aug_qa_mean_embed = aug_qa_mean_embed + pid_embed * (
                        aug_qa_embed_diff + q_embed_diff
                    )
                    aug_qa_cov_embed = aug_qa_cov_embed + pid_embed * (
                        aug_qa_embed_diff + q_embed_diff
                    )

            c_reg_loss = (pid_embed**2).sum() * self.l2

        # Transformer 编码
        if do_cl:
            cat_q_mean = torch.cat([q_mean_embed, aug_q_mean_embed], dim=0)
            cat_q_cov = torch.cat([q_cov_embed, aug_q_cov_embed], dim=0)
            cat_qa_mean = torch.cat([qa_mean_embed, aug_qa_mean_embed], dim=0)
            cat_qa_cov = torch.cat([qa_cov_embed, aug_qa_cov_embed], dim=0)

            cat_mean_out, cat_cov_out = self.model(
                cat_q_mean, cat_q_cov, cat_qa_mean, cat_qa_cov, self.atten_type
            )

            B = q_mean_embed.size(0)
            mean_output = cat_mean_out[:B]
            cov_output = cat_cov_out[:B]
            aug_mean_output = cat_mean_out[B:]
            aug_cov_output = cat_cov_out[B:]
        else:
            mean_output, cov_output = self.model(
                q_mean_embed, q_cov_embed, qa_mean_embed, qa_cov_embed, self.atten_type
            )

        # 对比学习损失
        if do_cl:
            if mask is not None:
                # 匹配原版: position 0 始终参与池化, positions 1..L-1 仅当 pair-validity 为 True 时参与
                adj_pair = mask[:, :-1] & mask[:, 1:]
                pool_mask = torch.cat(
                    [
                        torch.ones(mask.size(0), 1, 1, device=mask.device),
                        adj_pair.unsqueeze(-1).float(),
                    ],
                    dim=1,
                )
            else:
                pool_mask = torch.ones_like(mean_output[:, :, :1])

            pooled_mean = torch.mean(mean_output * pool_mask, dim=1)
            pooled_cov = torch.mean(cov_output * pool_mask, dim=1)
            pooled_aug_mean = torch.mean(aug_mean_output * pool_mask, dim=1)
            pooled_aug_cov = torch.mean(aug_cov_output * pool_mask, dim=1)

            if self.emb_type == "stoc_qid":
                cl_loss = self.wloss(
                    pooled_mean, pooled_cov, pooled_aug_mean, pooled_aug_cov
                )
            else:
                cl_loss = self.wloss(
                    pooled_mean, pooled_mean, pooled_aug_mean, pooled_aug_mean
                )

        # 预测
        if self.emb_type == "stoc_qid":
            concat_q = torch.cat(
                [mean_output, cov_output, q_mean_embed, q_cov_embed], dim=-1
            )
        else:
            concat_q = torch.cat(
                [mean_output, mean_output, q_cov_embed, q_cov_embed], dim=-1
            )

        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        # 协方差统计
        cov_stats = torch.mean(torch.mean(F.elu(cov_output) + 1, dim=-1), -1)

        return preds, cl_loss, cov_stats, c_reg_loss
