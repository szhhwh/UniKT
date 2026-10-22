"""HDKT 模型定义。

两级去噪的 LPKT 变体：在 LPKT 式学习过程追踪循环之前，用两组件生成
逐步硬去噪信号——
    1. 目标练习判别器：双向 LSTM 编码（题目嵌入 ⊕ 知识嵌入），与学生嵌入
       做 additive attention，hard gumbel 输出"该步是否目标练习"；
    2. 序列级 VAE：重构同一编码序列，hard gumbel 输出"该步是否可重构"；
两信号同时为 1 的位置视为噪声，将题目 / 答题用时 / 间隔时间三个嵌入置零
后再进 KT 循环（答案特征与技能指示向量不受影响）。总损失 = BCE + VAE
重构损失。

same-position 约定：``pred[:, t]`` 预测 ``a[:, t]``；q0-shift 代替上一步
learning、分解实现的遗忘门避免物化 [B, Sk, 3h]。
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.layers.interaction import skills_to_multihot


class VAE(nn.Module):
    """序列级 VAE，重构编码序列并参与去噪打分。"""

    def __init__(self, input_dim: int, h_dim: int, z_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class HDKTNet(nn.Module):
    """HDKT 网络：两级去噪前端 + LPKT 式 KT 循环。

    Args:
        num_questions: 题目数。
        num_skills: 知识点数。
        n_at: 答题用时词表大小（含 padding 0 行）。
        n_it: 间隔时间词表大小（含 padding 0 行）。
        num_users: 用户（序列行）数，学生嵌入词表。
        max_seq_len: 序列长度（去噪卷积的通道维，须与 batch 的实际 S 一致：
            数据 dense 填充、无动态 batching，S 恒等于 max_seq_len）。
        hidden_size: 隐藏维度 h（统一题目 / 时间 / 知识嵌入维度）。
        dropout: 门控 dropout 概率（learn_gains 与判别器 attention 共用）。
        tau: hard gumbel-softmax 温度。
        emb_dropout: 去噪组件的嵌入 dropout 概率。
        q_gamma: Smoothing added to each interaction's skill indicator.
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        n_at: int,
        n_it: int,
        num_users: int,
        max_seq_len: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
        tau: float = 100.0,
        emb_dropout: float = 0.3,
        q_gamma: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_skills = num_skills
        self.tau = tau
        h = hidden_size

        # ===== KT main body (LPKT-style) =====
        # id 0 of at/it is the zero padding row.
        self.e_embed = nn.Embedding(num_questions, h)
        self.at_embed = nn.Embedding(n_at, h, padding_idx=0)
        self.it_embed = nn.Embedding(n_it, h, padding_idx=0)

        self.linear_input = nn.Linear(3 * h, h)
        self.w_l = nn.Linear(4 * h, h)
        self.w_c = nn.Linear(4 * h, h)
        self.w_f = nn.Linear(3 * h, h)  # applied decomposed over [K, lg, it]
        self.linear_predict = nn.Linear(2 * h, h)

        # Trainable initial knowledge matrix, expanded per batch.
        self.k_matrix = nn.Parameter(torch.empty(num_skills, h))

        self.dropout = nn.Dropout(dropout)

        self.q_gamma = q_gamma

        # ===== Denoising front end =====
        self.student_emb = nn.Embedding(num_users, h)
        # Index = skill + num_skills * masked_label -> vocab 2 * num_skills,
        # row 0 = padding.
        self.knowledge_emb = nn.Embedding(2 * num_skills, h, padding_idx=0)
        # Shared by the discriminator and the seq-level VAE.
        self.rnn1 = nn.LSTM(2 * h, 2 * h, bidirectional=True, batch_first=True)
        self.attention_linear1 = nn.Linear(h, h)
        self.attention_linear2 = nn.Linear(2 * h, h)  # applied to enc_emb and k_
        self.attention_linear3 = nn.Linear(2 * h, 2)
        self.conv = nn.Conv2d(max_seq_len, max_seq_len, (1, 2))
        self.seq_level_mlp = nn.Sequential(
            nn.Linear(2 * h, 2, bias=False), nn.Sigmoid()
        )
        self.vae = VAE(2 * h, h, h)
        self.dropout1 = nn.Dropout(dropout)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier for embeddings/linears, zero for biases and padding rows."""
        for m in (
            self.e_embed,
            self.at_embed,
            self.it_embed,
            self.student_emb,
            self.knowledge_emb,
        ):
            nn.init.xavier_uniform_(m.weight)
        for m in (
            self.linear_input,
            self.w_l,
            self.w_c,
            self.w_f,
            self.linear_predict,
            self.attention_linear1,
            self.attention_linear2,
            self.attention_linear3,
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.k_matrix)
        # Restore the zero padding rows clobbered by xavier.
        with torch.no_grad():
            self.at_embed.weight[0].zero_()
            self.it_embed.weight[0].zero_()
            self.knowledge_emb.weight[0].zero_()

    def forward(
        self,
        e_data: torch.Tensor,
        at_data: torch.Tensor,
        a_data: torch.Tensor,
        it_data: torch.Tensor,
        mask: torch.Tensor,
        student: torch.Tensor,
        skill_data: torch.Tensor,
        skills: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        约定（same-position）：``pred[:, t]`` 预测 ``a_data[:, t]``，使用
        0..t-1 的信息；``pred[:, 0]`` 恒为 0，由提取阶段丢弃。

        Args:
            e_data: 题目序列 [B, S]
            at_data: 答题用时词表 id 序列 [B, S]
            a_data: 答案序列 [B, S]（输入特征与预测目标）
            it_data: 间隔时间词表 id 序列 [B, S]
            mask: 有效位置掩码 [B, S]
            student: 用户 id [B]（学生嵌入，进判别器）
            skill_data: 知识点序列 [B, S]（每题一个关联 skill，padding 为 0）
            skills: 本次交互的完整技能列表 [B, S, K]，以 num_skills 填充

        Returns:
            (pred, reloss): 预测 [B, S]，每个位置 ∈ (0, 1)；VAE 重构损失标量。
        """
        bs, seq_len = e_data.size(0), e_data.size(1)
        h = self.hidden_size

        # --- Denoising front end ---
        masked_label = a_data * mask.long()
        k_ = torch.cat(
            (
                self.e_embed(e_data),
                self.knowledge_emb(skill_data + self.num_skills * masked_label),
            ),
            dim=2,
        )  # [B, S, 2h]
        stu_signal = self._discriminator(self.student_emb(student), k_, mask)
        reloss, seq_score = self._seq_level_vae(k_, mask)
        # Steps flagged by BOTH gates are treated as noise; padding stays 0
        # via the mask.
        deno = (1.0 - seq_score * stu_signal) * mask.to(k_.dtype)  # [B, S]
        deno = deno.unsqueeze(-1)  # broadcast over h

        # Answers and skill indicators retain their observed values.
        e_embed_data = self.e_embed(e_data) * deno
        at_embed_data = self.at_embed(at_data) * deno
        it_embed_data = self.it_embed(it_data) * deno

        a_expand = a_data.unsqueeze(-1).expand(-1, -1, h).float()
        x = self.linear_input(
            torch.cat((e_embed_data, at_embed_data, a_expand), dim=2)
        )  # [B, S, h]

        # q0[t] = x[t-1], zero vector at t=0.
        q0 = torch.cat((x.new_zeros(bs, 1, h), x[:, :-1]), dim=1)

        K = self.k_matrix.unsqueeze(0).expand(bs, -1, -1)  # [B, Sk, h]

        pred = torch.zeros(bs, seq_len, device=e_data.device)
        skill_indicator = (
            skills_to_multihot(skills, self.num_skills).to(K.dtype) + self.q_gamma
        )

        for t in range(seq_len):
            kc_row = skill_indicator[:, t]
            kkk = torch.einsum("bs,bsh->bh", kc_row, K)

            if t > 0:
                pred[:, t] = torch.sigmoid(
                    self.linear_predict(
                        torch.cat((kkk, e_embed_data[:, t]), dim=1)
                    ).mean(dim=1)
                )

            if t == seq_len - 1:
                break  # no future step to predict; skip the final update

            q = torch.cat((q0[:, t], it_embed_data[:, t], x[:, t], kkk), dim=1)
            lg = torch.sigmoid(self.w_l(q)) * (1 + torch.tanh(self.w_c(q))) / 2
            lg_proj = self.dropout(torch.einsum("bh,bs->bsh", lg, kc_row))

            for_get = self._forget_gate(K, lg, it_embed_data[:, t])
            K = lg_proj + K * (1 - for_get)

        return pred, reloss

    def _forget_gate(self, K, lg, it_e):
        """遗忘门 σ(W_f·[K ⊕ LG ⊕ it]) -> [B, Sk, h]。

        分解实现，与拼接 Linear 数学等价但避免物化 [B, Sk, 3h]。
        """
        w_k, w_lg, w_it = self.w_f.weight.chunk(3, dim=1)
        pre = K.matmul(w_k.t())
        pre = pre + lg.matmul(w_lg.t()).unsqueeze(1)
        pre = pre + it_e.matmul(w_it.t()).unsqueeze(1)
        return torch.sigmoid(pre + self.w_f.bias)

    def _discriminator(self, q, k_, mask):
        """目标练习判别器。

        Args:
            q: 学生嵌入 [B, h]
            k_: 拼接嵌入 [B, S, 2h]
            mask: 有效位置掩码 [B, S]

        Returns:
            [B, S]：hard one-hot 第 1 类（1 = 目标练习）。
        """
        h = self.hidden_size
        mask1 = mask.unsqueeze(2).to(k_.dtype)
        enc_bi, _ = self.rnn1(self.emb_dropout(k_) * mask1)
        # Sum the two directions of the 4h output back to 2h.
        enc_emb = enc_bi[:, :, : 2 * h] + enc_bi[:, :, 2 * h :]

        q_ = self.dropout1(
            self.attention_linear1(q.unsqueeze(1).expand(-1, k_.size(1), -1))
        )
        k_1 = self.dropout1(self.attention_linear2(enc_emb))
        k_2 = self.dropout1(self.attention_linear2(k_))
        alpha = torch.sigmoid(
            self.attention_linear3(
                torch.tanh(
                    torch.repeat_interleave(q_, 2, dim=2) + torch.cat((k_1, k_2), dim=2)
                )
            )
        )
        return F.gumbel_softmax(alpha, tau=self.tau, hard=True)[:, :, 1]

    def _seq_level_vae(self, k_, mask):
        """序列级 VAE 重构。

        与判别器共享 rnn1 / emb_dropout；两次独立前向在训练态因 dropout
        重采样而不同，勿缓存复用。

        Returns:
            (reloss, score): 重构 MSE(sum) + KL 标量；hard gumbel 打分 [B, S]。
        """
        h = self.hidden_size
        mask1 = mask.unsqueeze(2).to(k_.dtype)
        enc_bi, _ = self.rnn1(self.emb_dropout(k_) * mask1)
        enc_emb = enc_bi[:, :, : 2 * h] + enc_bi[:, :, 2 * h :]

        x_reconst, mu, log_var = self.vae(enc_emb)
        reconst_loss = F.mse_loss(x_reconst * mask1, enc_emb * mask1, reduction="sum")
        # KL over all positions (padding included).
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Fuse (reconstruction, encoding) on a trailing dim, collapsed by a
        # (1, 2) convolution whose channels are the sequence positions.
        fused = self.conv(torch.stack((x_reconst, enc_emb), dim=-1)).squeeze(-1)
        score = self.seq_level_mlp(F.relu(self.emb_dropout(fused)))
        score = score * mask.unsqueeze(2).to(score.dtype)
        gumbel = F.gumbel_softmax(score, tau=self.tau, hard=True)
        return reconst_loss + kl_div, gumbel[:, :, 1]
