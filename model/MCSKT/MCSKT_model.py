"""MCSKT (Mamba Contextual encoding + dynamic Sparse attention Knowledge Tracing)

论文: Zhang, Zhang & Liu, "An efficient knowledge tracing model via Mamba
Contextual Encoding and Dynamic Sparse Attention mechanism", Engineering
Applications of Artificial Intelligence, 2026.

四个模块（论文 Section 3）：
1. 两阶段特征嵌入（Eq.1 Rasch 题目嵌入 + Eq.2 遗忘特征融合 ỹ_t = [y_t ⊗ F·f_t ; f_t]）
2. 上下文感知表示（双编码器 Q-encoder / K-encoder，均为 Mamba 块 Eq.3）→ x̂, ŷ
3. 知识状态抽取（动态 k-sparse 多头注意力，Algorithm 1 + Eq.4-6）→ h_t
4. 预测（Eq.8 两层 ReLU MLP + sigmoid）

same_position 约定：out[t] 利用历史 0..t-1 与当前题目 x_t 预测 response[t]，
trainer 通过 ``same_position=True`` 归一化为 next-item 视图，避免标签泄漏。
"""

import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch import nn


class MambaBlock(nn.Module):
    """单个 Mamba 块（论文 Eq.3）。

    Eq.3 的双分支结构（m = S6(SiLU(Conv1D(Linear(x))))，n = SiLU(Linear(x))，
    x̂ = Linear(m ⊗ n)）即 ``mamba_ssm.Mamba`` 内部计算；外层补残差 + LayerNorm
    以支持堆叠。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(x)
        return self.norm(self.dropout(y) + x)


class MambaEncoder(nn.Module):
    """堆叠若干 Mamba 块的编码器（Q-encoder 或 K-encoder）。"""

    def __init__(
        self,
        d_model: int,
        n_blocks: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MambaBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ForgettingFeatureEmbed(nn.Module):
    """遗忘特征向量 f_t（论文 Eq.2）。

    三类遗忘特征（repeated time gap / sequence time gap / past trial count，
    均 log2 离散化）的 one-hot 拼接经单层线性投影到 d_model。
    """

    def __init__(self, num_rgap: int, num_sgap: int, num_pcount: int, d_model: int):
        super().__init__()
        self.num_rgap = num_rgap
        self.num_sgap = num_sgap
        self.num_pcount = num_pcount
        self.proj = nn.Linear(num_rgap + num_sgap + num_pcount, d_model, bias=False)

    def forward(
        self, rgap: torch.Tensor, sgap: torch.Tensor, pcount: torch.Tensor
    ) -> torch.Tensor:
        r = F.one_hot(rgap.clamp(0, self.num_rgap - 1), self.num_rgap).float()
        s = F.one_hot(sgap.clamp(0, self.num_sgap - 1), self.num_sgap).float()
        p = F.one_hot(pcount.clamp(0, self.num_pcount - 1), self.num_pcount).float()
        return self.proj(torch.cat([r, s, p], dim=-1))


class DynamicKSparseAttention(nn.Module):
    """动态 k-sparse 多头注意力（论文 Algorithm 1 + Eq.4-6）。

    same_position 因果：query = x̂_t，keys = x̂_{<t}，values = ŷ_{<t}。
    - 时间衰减（Eq.4）：score ← (Q·K / √d) · exp(-θ·(t-j))，θ 每头可学习。
    - 动态 Top-K（Eq.5）：每头保留 Top-K（k = k_ratio·S，k_ratio ∈ [Δ1,Δ2]），
      其余置 -∞ 后 Softmax 重归一化。
    - 加权求和（Eq.6）：h_t = Σ Â_{t,j} · V_j。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        delta1: float = 0.25,
        delta2: float = 0.667,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.delta1 = delta1
        self.delta2 = delta2

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # 每头独立可学习衰减率 θ（softplus 保证 > 0）
        self.theta_raw = nn.Parameter(torch.full((num_heads,), -2.0))
        # 每头独立可学习稀疏比例 logit，sigmoid 映射到 [Δ1, Δ2]（初值取区间中点）
        self.k_logit = nn.Parameter(torch.zeros(num_heads))

    def compute(
        self,
        x_hat: torch.Tensor,
        y_hat: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x_hat.shape
        H, dh = self.num_heads, self.d_head

        # Eq.4: Q = x̂ W^Q, K = x̂ W^K, V = ŷ W^V
        q = self.q_lin(x_hat).view(B, S, H, dh).transpose(1, 2)
        k = self.k_lin(x_hat).view(B, S, H, dh).transpose(1, 2)
        v = self.v_lin(y_hat).view(B, S, H, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (dh**0.5)

        # 因果掩码（query t 仅看 key j<t）+ key padding 掩码
        causal = torch.triu(
            torch.ones(S, S, device=x_hat.device, dtype=torch.bool), diagonal=0
        )
        invalid = causal[None, None, :, :] | ~key_mask[:, None, None, :]

        # 时间衰减（Eq.4）：dist[t,j] = t - j（j<t 时为正）
        pos = torch.arange(S, device=x_hat.device, dtype=scores.dtype)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0.0)
        theta = F.softplus(self.theta_raw).view(1, H, 1, 1)
        scores = scores * torch.exp(-theta * dist[None, None, :, :])
        scores = scores.masked_fill(invalid, float("-inf"))

        # 动态 Top-K（Eq.5）：每头 k = round(k_ratio·S)，k_ratio ∈ [Δ1,Δ2]
        k_ratio = self.delta1 + (self.delta2 - self.delta1) * torch.sigmoid(
            self.k_logit
        )
        k_per_head = (k_ratio * S).round().clamp(min=1, max=S).long()
        k_max = int(k_per_head.max().item())
        head_keep = (
            torch.arange(k_max, device=x_hat.device)[None, :] < k_per_head[:, None]
        )

        topk_vals, topk_idx = scores.topk(k_max, dim=-1)
        del topk_vals
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, topk_idx, head_keep[None, :, None, :].expand_as(topk_idx))
        sparse_scores = scores.masked_fill(~keep, float("-inf"))
        del keep, topk_idx

        # 重归一化；全 -∞ 行（如 t=0 无历史）置 0
        attn = torch.nan_to_num(torch.softmax(sparse_scores, dim=-1), nan=0.0)

        # Eq.6: 加权求和 + 残差（query x̂，题目侧无 response，不会泄漏）+ LayerNorm
        out = (
            torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        )
        return self.norm(self.out_lin(out) + x_hat)


class MCSKT(nn.Module):
    """MCSKT 模型（论文 Section 3）。

    Args:
        num_c: 知识概念（技能）数量
        n_pid: 题目数量，>0 时启用 Rasch 嵌入
        num_rgap / num_sgap / num_pcount: 三类遗忘特征的桶数量
        d_model: 隐藏维度
        n_blocks: 每个编码器的 Mamba 块数量
        num_heads: 动态 k-sparse 注意力头数
        d_state / d_conv / expand: Mamba 块超参
        dropout: Dropout 概率
        l2: Rasch 难度参数 μ 的 L2 正则系数
        delta1 / delta2: 动态稀疏比例 k 的可学习区间 [Δ1, Δ2]（论文最佳 [1/4, 2/3]）
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        num_rgap: int = 100,
        num_sgap: int = 100,
        num_pcount: int = 15,
        d_model: int = 256,
        n_blocks: int = 5,
        num_heads: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        l2: float = 1e-5,
        delta1: float = 0.25,
        delta2: float = 0.667,
    ):
        super().__init__()
        self.model_name = "mcskt"
        self.num_c = num_c
        self.n_pid = n_pid
        self.l2 = l2
        d = d_model

        # Eq.1 两阶段特征嵌入（第一阶段：Rasch）
        self.q_embed = nn.Embedding(num_c, d)  # c_{k_t}
        self.qa_embed = nn.Embedding(2 * num_c + 1, d)  # e_{(k_t, r_t)}
        if n_pid > 0:
            self.difficult = nn.Embedding(n_pid + 1, 1)  # μ_{q_t}
            self.q_embed_diff = nn.Embedding(num_c, d)  # d_{k_t}
            self.qa_embed_diff = nn.Embedding(2 * num_c + 1, d)  # f_{(k_t, r_t)}
            nn.init.constant_(self.difficult.weight, 0.0)

        # Eq.2 遗忘特征融合
        self.forget_embed = ForgettingFeatureEmbed(num_rgap, num_sgap, num_pcount, d)
        self.f_transform = nn.Linear(d, d)  # F
        self.k_in_proj = nn.Linear(2 * d, d)  # ỹ = [y⊗F·f ; f] → d

        # Eq.3 双编码器
        self.q_encoder = MambaEncoder(d, n_blocks, d_state, d_conv, expand, dropout)
        self.k_encoder = MambaEncoder(d, n_blocks, d_state, d_conv, expand, dropout)

        # Algorithm 1 动态 k-sparse 注意力
        self.attn = DynamicKSparseAttention(d, num_heads, delta1, delta2)

        # Eq.8 预测：f1=ReLU(W1(h⊕x)), f2=ReLU(W2 f1), r̂=σ(W3 f2)
        self.pred1 = nn.Linear(2 * d, d)
        self.pred2 = nn.Linear(d, d)
        self.pred3 = nn.Linear(d, 1)
        self.dropout = nn.Dropout(dropout)

    def embed(
        self, sequence: torch.Tensor, response: torch.Tensor, pid_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """题目嵌入 x 与交互嵌入 y（论文 Eq.1）。"""
        c_reg = torch.tensor(0.0, device=sequence.device)
        qa = sequence + self.num_c * response  # concept-response 联合索引

        x = self.q_embed(sequence)
        y = self.qa_embed(qa)

        if self.n_pid > 0 and pid_data is not None:
            mu = self.difficult(pid_data)
            x = x + mu * self.q_embed_diff(sequence)
            y = y + mu * self.qa_embed_diff(qa)
            c_reg = (mu**2).sum() * self.l2

        return x, y, c_reg

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        pid_data: torch.Tensor,
        rgap: torch.Tensor,
        sgap: torch.Tensor,
        pcount: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播（same_position：out[t] 预测 response[t]）。

        Returns:
            preds: 预测概率 [B, S]
            c_reg_loss: Rasch 正则化损失
        """
        x, y, c_reg = self.embed(sequence, response, pid_data)
        x = self.dropout(x)
        y = self.dropout(y)

        # Eq.2 遗忘特征融合：ỹ_t = [y_t ⊗ F·f_t ; f_t]
        f = self.forget_embed(rgap, sgap, pcount)
        y_tilde = torch.cat([y * self.f_transform(f), f], dim=-1)
        k_input = self.k_in_proj(y_tilde)

        # Eq.3 双编码器
        x_hat = self.q_encoder(x)
        y_hat = self.k_encoder(k_input)

        # Algorithm 1 动态 k-sparse 注意力 → 知识状态 h_t
        h = self.attn.compute(x_hat, y_hat, mask.bool())

        # Eq.8 预测
        p = torch.cat([h, x], dim=-1)
        p = self.dropout(F.relu(self.pred1(p)))
        p = self.dropout(F.relu(self.pred2(p)))
        preds = torch.sigmoid(self.pred3(p).squeeze(-1))
        return preds, c_reg


__all__ = ["MCSKT"]
