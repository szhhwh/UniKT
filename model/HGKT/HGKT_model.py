"""HGKT 模型定义。

实现自论文：
    Ma et al., "Hypergraph-Driven High-Order Knowledge Tracing with a
    Dual-Gated Dynamic Mechanism", Appl. Sci. 2025, 15, 8617.

逐步递归（same-position：``pred[:, t]`` 预测 ``a[:, t]``，使用 0..t-1 信息）：
    E       = HyperEnc(e_embed)            超图增强练习嵌入 [Q, dk]
    x[t]    = W3·[E(e_t) ⊕ at_t ⊕ a_t]     学习单元表示
    lg      = σ(W5·q)·(1 + tanh(W4·q))/2   q = [x[t] ⊕ it_t ⊕ x[t-1] ⊕ kkk]
    lg_proj = dropout(kc(t) ⊗ lg)          RLG_t = q_et ⊙ LG_t 的 K-矩阵化
    Fe      = σ(W6·[K ⊕ LG ⊕ it_t])        遗忘擦除门
    Fa      = tanh(W7·[K ⊕ LG ⊕ it_t])     遗忘更新门
    F       = min((1−Fe)(1+Fa), 1)          双门遗忘，投影到 (0, 1]
    K       ← lg_proj + K ⊙ F
    pred[t] = σ(mean(W8·[kc(t)@K ⊕ e_raw(t)]))
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from model.layers.interaction import skills_to_multihot


class _HypergraphConvolution(nn.Module):
    """单向超图卷积层：``factor @ (x W) + b``"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        stdv = 1.0 / math.sqrt(out_features)
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(factor, x.matmul(self.weight)) + self.bias


class _HypergraphEncoder(nn.Module):
    """两层超图编码：exercise→concept（ReLU）→exercise"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.hgc1 = _HypergraphConvolution(in_dim, hidden_dim)
        self.hgc2 = _HypergraphConvolution(hidden_dim, out_dim)

    def forward(
        self, x: torch.Tensor, hyper_q: torch.Tensor, hyper_p: torch.Tensor
    ) -> torch.Tensor:
        x = F.relu(self.hgc1(x, hyper_q))  # [Q, de] -> [M, dk]
        return self.hgc2(x, hyper_p)  # [M, dk] -> [Q, dk]


class HGKTNet(nn.Module):
    """HGKT 网络（超图增强嵌入 + 学习增益 + 双门遗忘）。

    Args:
        num_questions: 题目数 Q。
        num_skills: 知识点（概念）数 M。
        n_at: 答题用时词表大小（含 padding 0 行）。
        n_it: 间隔时间词表大小（含 padding 0 行）。
        hyper_factors: 超图拉普拉斯因子 (P, Q_f)，P = Dv^-1/2·H·De^-1
            [Q, M]，Q_f = H^T·Dv^-1/2 [M, Q]。
        emb_dim: 原始练习嵌入维度 de（超图编码输入，预测层复用）。
        hidden_size: 知识状态与各门维度 dk。
        time_dim: 答题 / 间隔时间嵌入维度 da。
        dropout: dropout 概率（仅施加于 learn_gains）。
        q_gamma: Smoothing added to each interaction's skill indicator.
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        n_at: int,
        n_it: int,
        hyper_factors: tuple[torch.Tensor, torch.Tensor],
        emb_dim: int = 128,
        hidden_size: int = 128,
        time_dim: int = 50,
        dropout: float = 0.2,
        q_gamma: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_skills = num_skills
        self.time_dim = time_dim
        de, dk, da = emb_dim, hidden_size, time_dim

        # id 0 of at/it is the zero padding row.
        self.e_embed = nn.Embedding(num_questions, de)
        self.at_embed = nn.Embedding(n_at, da, padding_idx=0)
        self.it_embed = nn.Embedding(n_it, da, padding_idx=0)

        self.hgnn = _HypergraphEncoder(de, dk, dk)
        self.linear_input = nn.Linear(dk + 2 * da, dk)

        # W4 ↔ w_c (tanh), W5 ↔ w_l (σ); gate input [l_t ⊕ it_t ⊕ l_{t-1} ⊕ h_{t-1}]
        self.w_l = nn.Linear(3 * dk + da, dk)
        self.w_c = nn.Linear(3 * dk + da, dk)

        self.w_fe = nn.Linear(2 * dk + da, dk)  # σ branch
        self.w_fa = nn.Linear(2 * dk + da, dk)  # tanh branch

        self.linear_predict = nn.Linear(dk + de, dk)

        # Trainable initial knowledge matrix, expanded per batch.
        self.k_matrix = nn.Parameter(torch.empty(num_skills, dk))

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Static graph is deterministically rebuilt from data each run.
        hyper_p, hyper_q = hyper_factors
        self.register_buffer("hyper_p", hyper_p, persistent=False)
        self.register_buffer("hyper_q", hyper_q, persistent=False)
        self.q_gamma = q_gamma

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier for embeddings and dense layers, zero for biases."""
        for m in (self.e_embed, self.at_embed, self.it_embed):
            nn.init.xavier_uniform_(m.weight)
        for m in (
            self.linear_input,
            self.w_l,
            self.w_c,
            self.linear_predict,
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # Forget-gate init anchors Fe high (σ(2) ≈ 0.88) so the non-contractive
        # factor F = (1−Fe)(1+Fa) ≤ 2(1−Fe) starts below 1.
        nn.init.xavier_uniform_(self.w_fe.weight, gain=0.1)
        nn.init.xavier_uniform_(self.w_fa.weight, gain=0.1)
        nn.init.zeros_(self.w_fa.bias)
        nn.init.constant_(self.w_fe.bias, 2.0)
        # Restore the zero padding rows clobbered by xavier.
        with torch.no_grad():
            self.at_embed.weight[0].zero_()
            self.it_embed.weight[0].zero_()
        nn.init.xavier_uniform_(self.k_matrix)

    def _forget_pre_pair(
        self,
        K: torch.Tensor,
        lg: torch.Tensor,
        it_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """两门预激活 (pre_fe, pre_fa) = W·[K ⊕ LG ⊕ it] 的分解实现。"""
        dk = self.hidden_size
        w_k = torch.cat(
            (self.w_fe.weight[:, :dk], self.w_fa.weight[:, :dk]), dim=0
        )  # [2dk, dk]
        pre = K.matmul(w_k.t())  # [B, M, 2dk]
        lg_terms = torch.cat(
            (
                lg.matmul(self.w_fe.weight[:, dk : 2 * dk].t()),
                lg.matmul(self.w_fa.weight[:, dk : 2 * dk].t()),
            ),
            dim=1,
        ).unsqueeze(1)
        it_terms = torch.cat(
            (
                it_e.matmul(self.w_fe.weight[:, 2 * dk :].t()),
                it_e.matmul(self.w_fa.weight[:, 2 * dk :].t()),
            ),
            dim=1,
        ).unsqueeze(1)
        bias = torch.cat((self.w_fe.bias, self.w_fa.bias))
        pre = pre + lg_terms + it_terms + bias
        return pre[..., :dk], pre[..., dk:]

    def forward(
        self,
        e_data: torch.Tensor,
        at_data: torch.Tensor,
        a_data: torch.Tensor,
        it_data: torch.Tensor,
        skills: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            e_data: 题目序列 [B, S]
            at_data: 答题用时词表 id 序列 [B, S]
            a_data: 答案序列 [B, S]（输入特征与预测目标）
            it_data: 间隔时间词表 id 序列 [B, S]
            skills: 本次交互的技能列表 [B, S, K]，以 num_skills 填充

        Returns:
            pred: [B, S]，每个位置 ∈ (0, 1)；``pred[:, 0]`` 恒为 0，
            由提取阶段丢弃。
        """
        bs, seq_len = e_data.size(0), e_data.size(1)
        dk, da = self.hidden_size, self.time_dim
        device = e_data.device

        # Hypergraph encoding reruns every forward so gradients reach e_embed.
        enhanced = self.hgnn(self.e_embed.weight, self.hyper_q, self.hyper_p)
        e_enh_data = F.embedding(e_data, enhanced)  # [B, S, dk]
        e_raw_data = self.e_embed(e_data)  # [B, S, de], for the predict layer

        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)

        a_expand = a_data.unsqueeze(-1).expand(-1, -1, da).float()

        x = self.linear_input(
            torch.cat((e_enh_data, at_embed_data, a_expand), dim=2)
        )  # [B, S, dk]

        # q0[t] = x[t-1], zero vector at t=0.
        q0 = torch.cat((x.new_zeros(bs, 1, dk), x[:, :-1]), dim=1)

        K = self.k_matrix.unsqueeze(0).expand(bs, -1, -1)  # [B, M, dk]

        kc_rows = skills_to_multihot(skills, self.num_skills).to(K.dtype) + self.q_gamma
        # pred[:, 0] is a zero placeholder, dropped by the extraction stage.
        preds = [torch.zeros(bs, device=device)]

        for t in range(seq_len):
            kc_row = kc_rows[:, t]
            # Prediction and learning gate share the same state and kc row,
            # so a single aggregation serves both.
            kkk = torch.einsum("bm,bmd->bd", kc_row, K)  # [B, dk]

            if t > 0:
                z = self.linear_predict(torch.cat((kkk, e_raw_data[:, t]), dim=1))
                preds.append(self.sigmoid(z.mean(dim=1)))

            if t == seq_len - 1:
                break  # no future step to predict; skip the final update

            it_e = it_embed_data[:, t]
            q = torch.cat((x[:, t], it_e, q0[:, t], kkk), dim=1)  # [B, 3dk+da]
            lg = self.sigmoid(self.w_l(q))
            lg = lg * (1 + self.tanh(self.w_c(q))) / 2  # [B, dk]
            lg_proj = self.dropout(torch.einsum("bd,bm->bmd", lg, kc_row))

            pre_fe, pre_fa = self._forget_pre_pair(K, lg, it_e)
            fe = self.sigmoid(pre_fe)
            fa = self.tanh(pre_fa)
            # Paper leaves F ∈ (0, 2); the F > 1 half diverges over the
            # 200-step recurrence, so F is clamped to the (0, 1] decay spectrum.
            forget = ((1 - fe) * (1 + fa)).clamp(max=1.0)  # [B, M, dk]
            K = lg_proj + K * forget

        return torch.stack(preds, dim=1)
