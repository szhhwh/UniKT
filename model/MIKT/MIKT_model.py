"""MIKT (Multiscale State Representation Knowledge Tracing) 模型实现

原始论文: "Interpretable Knowledge Tracing with Multiscale State Representation", WWW 2024
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.interaction import skills_to_multihot


class MIKT(nn.Module):
    """MIKT 模型

    通过多尺度状态表示（技能级状态 + 全局状态）实现可解释的知识追踪，
    使用时间间隔遗忘机制和 IRT 风格的预测公式。

    Args:
        embed_dim: 嵌入维度（需等于 state_dim）
        state_dim: 状态表示维度
        dropout: Dropout 概率
        data_metadata: 数据集元数据，需包含 num_questions, num_skills, max_seq_len
    """

    def __init__(self, embed_dim, state_dim, dropout, data_metadata, **kwargs):
        super().__init__()

        pro_max = data_metadata["num_questions"]
        skill_max = data_metadata["num_skills"]
        max_seq = data_metadata["max_question_seq_len"]

        d = embed_dim
        state_d = state_dim
        p = dropout
        assert d == state_d, (
            f"MIKT requires embed_dim == state_dim, got {d} != {state_d}"
        )

        self.pro_max = pro_max
        self.skill_max = skill_max
        self.max_seq = max_seq

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))
        nn.init.xavier_uniform_(self.pro_embed)

        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))
        nn.init.xavier_uniform_(self.skill_embed)

        self.skill_state = nn.Parameter(torch.rand(skill_max, state_d))
        self.time_state = nn.Parameter(torch.rand(max_seq, state_d))
        self.all_state = nn.Parameter(torch.rand(1, state_d))

        self.all_forget = nn.Sequential(
            nn.Linear(2 * state_d, state_d),
            nn.ReLU(),
            nn.Linear(state_d, state_d),
            nn.Sigmoid(),
        )

        self.ans_embed = nn.Embedding(2, d)

        self.now_obtain = nn.Sequential(
            nn.Linear(d, state_d),
            nn.Tanh(),
            nn.Linear(state_d, state_d),
            nn.Tanh(),
        )

        self.pro_diff = nn.Embedding(pro_max, 1)

        self.pro_linear = nn.Linear(d, d)
        self.skill_linear = nn.Linear(d, d)
        self.pro_change = nn.Linear(d, d)

        self.pro_ability = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self.all_obtain = nn.Linear(d, d)

        self.skill_forget = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d),
        )

        self.predict_attn = nn.Linear(3 * d, d)

        self.dropout = nn.Dropout(p=p)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, question, response, mask, skills):
        """前向传播

        Args:
            question: 题目ID序列 [B, S]
            response: 答题结果序列 [B, S]
            mask: 有效位置掩码 [B, S]
            skills: 本次交互的技能列表 [B, S, K]，以 num_skills 填充

        Returns:
            predictions: 预测概率 [B, S-1]
        """
        device = question.device

        next_problem = question[:, 1:]
        next_ans = response[:, 1:]

        seq = next_problem.shape[1]
        batch = question.shape[0]

        pro_embed = self.pro_embed
        skill_embed = self.skill_embed

        skill_indicator = skills_to_multihot(skills, self.skill_max).to(
            skill_embed.dtype
        )
        skill_mean = torch.matmul(skill_indicator, skill_embed) / (
            skill_indicator.sum(dim=-1, keepdim=True).clamp_min(1)
        )
        pro_idx = torch.arange(self.pro_max, device=device)
        pro_diff = torch.sigmoid(self.pro_diff(pro_idx))
        q_pro = self.pro_linear(F.embedding(question, pro_embed))
        q_skill = self.skill_linear(skill_embed)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(
            q_pro.shape[-1]
        )
        attn = torch.softmax(attn.masked_fill(skill_indicator == 0, -1e9), dim=-1)
        skill_attn = torch.matmul(attn, skill_embed)
        now_embed = skill_attn + F.embedding(question, pro_diff) * self.pro_change(
            skill_mean
        )
        next_pro_rasch = self.dropout(now_embed[:, 1:])
        next_X = next_pro_rasch + self.ans_embed(next_ans)
        last_all_time = torch.ones(batch, device=device).long()

        time_embed = self.time_state
        all_gap_embed = F.embedding(last_all_time, time_embed)

        res_p = []

        last_skill_time = torch.zeros(batch, self.skill_max, device=device).long()
        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)

        all_state = self.all_state.repeat(batch, 1)

        for now_step in range(seq):
            now_pro = next_problem[:, now_step]
            now_pro2skill = skill_indicator[:, now_step + 1].unsqueeze(1)

            now_pro_embed = next_pro_rasch[:, now_step]

            f1 = now_pro_embed.unsqueeze(1)
            f2 = skill_state

            skill_time_gap = now_step - now_pro2skill.squeeze(1) * last_skill_time
            skill_time_gap_embed = F.embedding(
                skill_time_gap.clamp(0, self.max_seq - 1).long(), time_embed
            )

            now_all_state = all_state

            forget_now_all_state = now_all_state * self.all_forget(
                self.dropout(torch.cat([now_all_state, all_gap_embed], dim=-1))
            )

            effect_all_state = forget_now_all_state.unsqueeze(1).repeat(
                1, f2.shape[1], 1
            )

            skill_forget_val = torch.sigmoid(
                self.skill_forget(
                    self.dropout(
                        torch.cat(
                            [skill_state, skill_time_gap_embed, effect_all_state],
                            dim=-1,
                        )
                    )
                )
            )
            skill_forget_val = torch.masked_fill(
                skill_forget_val, now_pro2skill.transpose(-1, -2) == 0, 1
            )
            skill_state = skill_state * skill_forget_val

            now_pro_skill_attn = (
                torch.matmul(f1, skill_state.transpose(-1, -2)) / f1.shape[-1]
            )
            now_pro_skill_attn = torch.masked_fill(
                now_pro_skill_attn, now_pro2skill == 0, -1e9
            )
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)

            now_need_state = torch.matmul(now_pro_skill_attn, skill_state).squeeze(1)

            all_attn = torch.sigmoid(
                self.predict_attn(
                    self.dropout(
                        torch.cat(
                            [
                                now_need_state,
                                forget_now_all_state,
                                now_pro_embed,
                            ],
                            dim=-1,
                        )
                    )
                )
            )
            now_need_state = torch.cat(
                [(1 - all_attn) * now_need_state, all_attn * forget_now_all_state],
                dim=-1,
            )

            last_skill_time = torch.masked_fill(
                last_skill_time, now_pro2skill.squeeze(1) == 1, now_step
            )

            now_ability = torch.sigmoid(
                self.pro_ability(torch.cat([now_need_state, now_pro_embed], dim=-1))
            )
            now_diff = F.embedding(now_pro, pro_diff)

            now_output = torch.sigmoid(5 * (now_ability - now_diff))
            now_output = now_output.squeeze(-1)

            res_p.append(now_output)

            now_X = next_X[:, now_step]

            all_state = forget_now_all_state + torch.tanh(
                self.all_obtain(self.dropout(now_X))
            ).squeeze(1)

            to_get = torch.tanh(self.now_obtain(self.dropout(now_X))).unsqueeze(1)

            f1 = to_get
            f2 = skill_state

            now_pro_skill_attn = torch.matmul(f1, f2.transpose(-1, -2)) / f1.shape[-1]
            now_pro_skill_attn = torch.masked_fill(
                now_pro_skill_attn, now_pro2skill == 0, -1e9
            )
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)

            now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)
            skill_state = skill_state + now_get

        P = torch.vstack(res_p).T

        return P
