"""LPKT / LPKT-S 模型定义。

实现自论文：
    Shen et al., "Monitoring Student Progress for Learning Process-consistent
    Knowledge Tracing", IEEE TKDE 2022 (doi:10.1109/TKDE.2022.3221985)。

逐步递归（先用处理完 0..t-1 的知识矩阵 K 预测 a[t]，再用交互 t 更新 K）：
    x[t]    = Linear([problem_emb ⊕ at_emb ⊕ answer_expand])
    q       = [q0=x[t-1] ⊕ it[t] ⊕ x[t] ⊕ kkk]，kkk = kc(t) @ K
    lg      = σ(W_l · gate_input) · (1 + tanh(W_c · q)) / 2
    lg_proj = dropout(kc(t) ⊗ lg)
    for_get = σ(W_f · [K ⊕ LG ⊕ it (⊕ student)])
    K       ← lg_proj + K ⊙ (1 − for_get)
    pred[t] = σ(mean(Linear([kc(t) @ K ⊕ problem_emb(t) (⊕ student)])))

LPKT-S 在学习门 σ 支路、遗忘门、预测层前置三处注入 student embedding，
经模板方法 hook 下放给子类。
"""

import torch
from torch import nn

from model.layers.interaction import skills_to_multihot


class LPKTBase(nn.Module):
    """LPKT 家族共享结构（模板方法基类）。

    Args:
        num_questions: 题目数。
        num_skills: 知识点（概念）数。
        n_at: 答题用时词表大小（含 padding 0 行）。
        n_it: 间隔时间词表大小（含 padding 0 行）。
        hidden_size: 隐藏维度 h。
        dropout: dropout 概率（仅施加于 learn_gains）。
        q_gamma: Smoothing added to each interaction's skill indicator.
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        n_at: int,
        n_it: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
        q_gamma: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_skills = num_skills
        h = hidden_size

        # id 0 of at/it is the zero padding row.
        self.e_embed = nn.Embedding(num_questions, h)
        self.at_embed = nn.Embedding(n_at, h, padding_idx=0)
        self.it_embed = nn.Embedding(n_it, h, padding_idx=0)

        self.linear_input = nn.Linear(3 * h, h)

        # Both gate branches take 4h input; the σ branch input differs per
        # variant (_learning_gate_input), the tanh branch always takes q.
        self.w_l = nn.Linear(4 * h, h)
        self.w_c = nn.Linear(4 * h, h)

        # Trainable initial knowledge matrix, expanded per batch.
        self.k_matrix = nn.Parameter(torch.empty(num_skills, h))

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.q_gamma = q_gamma

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier for weights, zero for biases."""
        for m in (self.e_embed, self.at_embed, self.it_embed):
            nn.init.xavier_uniform_(m.weight)
        for m in (self.linear_input, self.w_l, self.w_c):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # Restore the zero padding rows clobbered by xavier.
        with torch.no_grad():
            self.at_embed.weight[0].zero_()
            self.it_embed.weight[0].zero_()
        nn.init.xavier_uniform_(self.k_matrix)

    # ===== Variant hooks (implemented by subclasses) =====

    def _learning_gate_input(
        self,
        q0: torch.Tensor,
        it_e: torch.Tensor,
        x: torch.Tensor,
        kkk: torch.Tensor,
        student: torch.Tensor | None,
    ) -> torch.Tensor:
        """σ 支路的输入 [B, 4h]；LPKT 用 q，LPKT-S 把 kkk 换成 student。"""
        raise NotImplementedError

    def _forget_gate(
        self,
        K: torch.Tensor,
        lg: torch.Tensor,
        it_e: torch.Tensor,
        student: torch.Tensor | None,
    ) -> torch.Tensor:
        """遗忘门 σ(W_f · [K ⊕ LG ⊕ it (⊕ student)]) -> [B, num_skills, h]。"""
        raise NotImplementedError

    def _predict_logits(
        self,
        k_proj: torch.Tensor,
        e_emb: torch.Tensor,
        student: torch.Tensor | None,
    ) -> torch.Tensor:
        """预测层 σ(mean(Linear([kc 投影 ⊕ 题目嵌入 (⊕ student)]))) -> [B]。"""
        raise NotImplementedError

    def forward(
        self,
        e_data: torch.Tensor,
        at_data: torch.Tensor,
        a_data: torch.Tensor,
        it_data: torch.Tensor,
        skills: torch.Tensor,
        student: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        约定（same-position）：``pred[:, t]`` 预测 ``a_data[:, t]``，使用
        0..t-1 的信息；``pred[:, 0]`` 恒为 0，由提取阶段丢弃。

        Args:
            e_data: 题目序列 [B, S]
            at_data: 答题用时词表 id 序列 [B, S]
            a_data: 答案序列 [B, S]（输入特征与预测目标）
            it_data: 间隔时间词表 id 序列 [B, S]
            skills: 本次交互的技能列表 [B, S, K]，以 num_skills 填充
            student: 用户 id [B]（LPKT-S 用，LPKT 忽略）

        Returns:
            pred: [B, S]，每个位置 ∈ (0, 1)。
        """
        bs, seq_len = e_data.size(0), e_data.size(1)
        h = self.hidden_size
        device = e_data.device

        e_embed_data = self.e_embed(e_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)

        a_expand = a_data.unsqueeze(-1).expand(-1, -1, h).float()

        x = self.linear_input(
            torch.cat((e_embed_data, at_embed_data, a_expand), dim=2)
        )  # [B, S, h]

        # q0[t] = x[t-1], zero vector at t=0.
        q0 = torch.cat((x.new_zeros(bs, 1, h), x[:, :-1]), dim=1)

        K = self.k_matrix.unsqueeze(0).expand(bs, -1, -1)  # [B, Sk, h]

        pred = torch.zeros(bs, seq_len, device=device)
        skill_indicator = (
            skills_to_multihot(skills, self.num_skills).to(K.dtype) + self.q_gamma
        )

        for t in range(seq_len):
            kc_row = skill_indicator[:, t]
            # Prediction and learning gate share the same state and kc row,
            # so a single aggregation serves both.
            kkk = torch.einsum("bs,bsh->bh", kc_row, K)  # [B, h]

            if t > 0:
                pred[:, t] = self._predict_logits(kkk, e_embed_data[:, t], student)

            if t == seq_len - 1:
                break  # no future step to predict; skip the final update

            q = torch.cat((q0[:, t], it_embed_data[:, t], x[:, t], kkk), dim=1)
            gate_in = self._learning_gate_input(
                q0[:, t], it_embed_data[:, t], x[:, t], kkk, student
            )
            lg = self.sigmoid(self.w_l(gate_in))
            lg = lg * (1 + self.tanh(self.w_c(q))) / 2  # [B, h]
            lg_proj = self.dropout(torch.einsum("bh,bs->bsh", lg, kc_row))

            for_get = self._forget_gate(K, lg, it_embed_data[:, t], student)
            K = lg_proj + K * (1 - for_get)

        return pred


class LPKTNet(LPKTBase):
    """LPKT（无 student embedding）：``w_f = Linear(3h, h)``、
    ``linear_predict = Linear(2h, h)``。"""

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        n_at: int,
        n_it: int,
        num_users: int | None = None,  # accepted for a uniform signature; unused
        hidden_size: int = 128,
        dropout: float = 0.2,
        q_gamma: float = 0.0,
    ):
        super().__init__(
            num_questions, num_skills, n_at, n_it, hidden_size, dropout, q_gamma
        )
        h = hidden_size
        self.w_f = nn.Linear(3 * h, h)
        self.linear_predict = nn.Linear(2 * h, h)
        nn.init.xavier_uniform_(self.w_f.weight)
        nn.init.zeros_(self.w_f.bias)
        nn.init.xavier_uniform_(self.linear_predict.weight)
        nn.init.zeros_(self.linear_predict.bias)

    def _learning_gate_input(self, q0, it_e, x, kkk, student):
        return torch.cat((q0, it_e, x, kkk), dim=1)  # same as q

    def _forget_gate(self, K, lg, it_e, student):
        # Decomposed σ(W_f·[K ⊕ LG ⊕ it] + b): mathematically identical to the
        # concatenated Linear but avoids materializing [B, Sk, 3h] (memory-critical).
        w_k, w_lg, w_it = self.w_f.weight.chunk(3, dim=1)
        pre = K.matmul(w_k.t())
        pre = pre + lg.matmul(w_lg.t()).unsqueeze(1)
        pre = pre + it_e.matmul(w_it.t()).unsqueeze(1)
        return torch.sigmoid(pre + self.w_f.bias)

    def _predict_logits(self, k_proj, e_emb, student):
        z = self.linear_predict(torch.cat((k_proj, e_emb), dim=1))
        return torch.sigmoid(z.mean(dim=1))


class LPKTSNet(LPKTBase):
    """LPKT-S（student embedding 三处注入）：``student_embed = Embedding(num_users, h)``、
    ``w_f = Linear(4h, h)``、``linear_predict = Linear(3h, h)``（student 前置拼接）。"""

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        n_at: int,
        n_it: int,
        num_users: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
        q_gamma: float = 0.0,
    ):
        super().__init__(
            num_questions, num_skills, n_at, n_it, hidden_size, dropout, q_gamma
        )
        h = hidden_size
        # Vocabulary covers ALL users; val/test embeddings stay at init.
        self.student_embed = nn.Embedding(num_users, h)
        self.w_f = nn.Linear(4 * h, h)
        self.linear_predict = nn.Linear(3 * h, h)
        nn.init.xavier_uniform_(self.student_embed.weight)
        nn.init.xavier_uniform_(self.w_f.weight)
        nn.init.zeros_(self.w_f.bias)
        nn.init.xavier_uniform_(self.linear_predict.weight)
        nn.init.zeros_(self.linear_predict.bias)

    def forward(self, e_data, at_data, a_data, it_data, skills, student=None):
        # Look up the student embedding once; hooks receive [B, h].
        s = self.student_embed(student)
        return super().forward(e_data, at_data, a_data, it_data, skills, s)

    def _learning_gate_input(self, q0, it_e, x, kkk, student):
        return torch.cat((q0, it_e, x, student), dim=1)

    def _forget_gate(self, K, lg, it_e, student):
        # Decomposed σ(W_f·[K ⊕ LG ⊕ it ⊕ student] + b); see LPKTNet._forget_gate.
        w_k, w_lg, w_it, w_s = self.w_f.weight.chunk(4, dim=1)
        pre = K.matmul(w_k.t())
        pre = pre + lg.matmul(w_lg.t()).unsqueeze(1)
        pre = pre + it_e.matmul(w_it.t()).unsqueeze(1)
        pre = pre + student.matmul(w_s.t()).unsqueeze(1)
        return torch.sigmoid(pre + self.w_f.bias)

    def _predict_logits(self, k_proj, e_emb, student):
        z = self.linear_predict(torch.cat((student, k_proj, e_emb), dim=1))
        return torch.sigmoid(z.mean(dim=1))
