"""TCKT 模型定义。

实现自论文：
    Huang et al., "Learning consistent representations with temporal and causal
    enhancement for knowledge tracing", Expert Systems With Applications 245
    (2024) 123128.

模型由四部分组成：
    1. 输入特征：交互嵌入 ``x_t = MLP([q_t ⊕ c_t ⊕ a_t ⊕ dif_t])``，其中难度 ``dif``
       由贝叶斯估计（论文式 2）得到。
    2. CSAT（因果自注意力）：LSAT（对当前序列做因果自注意力，估计 E[M|X][M]）+
       GSAT（以全局字典作为 K/V 做注意力，估计 E[X'][X']）→ 拼接 + FC（论文式
       14-16）。全局字典由交互嵌入的 K-Means 聚类在线生成（见 trainer）。
    3. LBS（学习行为模拟）：仿 LSTM 的遗忘门（使用间隔时间）+ 输入门与获得项
       （使用响应时间）逐时刻更新每个概念的知识状态（论文式 17-21）。
    4. 预测层：``y_{t+1} = σ(MLP([ĥ_t ⊕ q_{t+1}]))``（论文式 22）。

本实现修正原作者代码中的若干缺陷：
    - 源代码 forward 中引用了未定义的 ``recent_l``（以及未使用的 ``recent_c``、
      ``ca_data``），此处删除。
    - 全局字典原先从磁盘 ``./center_learning_2000.pt`` 加载，此处改为在线生成（见
      trainer）。
同时纠正了一处明显的拷贝错误：GSAT 的注意力掩码在参考代码中使用了
``triu(ones((seq_len, N)), k=1)``（把 LSAT 的时序因果掩码套用到无时序的全局字典型
上），这与论文式 15（对所有 x' 求期望）矛盾，因此 GSAT 不再施加掩码。
"""

import torch
from torch import nn

from model.layers.interaction import skills_to_multihot


class TCKTNet(nn.Module):
    """Temporal- and Causal-enhanced Knowledge Tracing 模型。

    Args:
        num_questions: 题目数（题目 id 为 0..num_questions-1）。
        num_skills: 知识点（概念）数。
        n_at: 响应时间（秒）的离散桶数。
        n_it: 间隔时间（分钟）的离散桶数。
        d_k: 隐藏维度。
        d_a: 答案 / 难度向量的展开维度。
        d_e: 题目嵌入维度。
        num_heads: 多头注意力的头数。
        seq_len: 固定序列长度（位置编码用）。
        global_dict_size: 全局字典大小 N（K-Means 簇数）。
        dropout: dropout 概率。
        q_gamma: Smoothing added to each interaction's skill indicator.
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        n_at: int,
        n_it: int,
        d_k: int = 128,
        d_a: int = 64,
        d_e: int = 128,
        num_heads: int = 8,
        seq_len: int = 200,
        global_dict_size: int = 400,
        dropout: float = 0.2,
        q_gamma: float = 0.0,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.num_skills = num_skills
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.global_dict_size = global_dict_size

        # 输入嵌入
        self.e_embed = nn.Embedding(num_questions, d_e)
        # 概念嵌入：0 号留给“无知识点”（多概念题取主知识点，无映射时为 0）。
        self.c_embed = nn.Embedding(num_skills + 1, d_k, padding_idx=0)
        self.at_embed = nn.Embedding(n_at, d_k)
        self.it_embed = nn.Embedding(n_it, d_k)

        # 交互嵌入 x_t = MLP([q_t ⊕ c_t ⊕ a_t ⊕ dif_t])（论文式 3）
        self.linear_all_learning = nn.Linear(d_e + d_k + d_a + d_a, d_k)

        # CSAT：局部 / 全局采样自注意力
        self.pos_embedding = nn.Embedding(seq_len, d_k)
        # LSAT：对当前序列的因果自注意力（论文式 14）
        self.linear_is = nn.ModuleList([nn.Linear(d_k, d_k) for _ in range(3)])
        self.multi_attention_is = nn.MultiheadAttention(
            embed_dim=d_k, num_heads=num_heads, dropout=dropout
        )
        self.layer_norm1_is = nn.LayerNorm(d_k)
        # GSAT：以全局字典作为 K/V 的注意力（论文式 15）
        self.linear_cs = nn.ModuleList([nn.Linear(d_k, d_k) for _ in range(3)])
        self.multi_attention_cs = nn.MultiheadAttention(
            embed_dim=d_k, num_heads=num_heads, dropout=dropout
        )
        self.layer_norm1_cs = nn.LayerNorm(d_k)
        # 融合 LSAT 与 GSAT（论文式 16）
        self.fc_att_cs_is = nn.Linear(2 * d_k, d_k)

        # LBS：学习行为模拟的门控参数（论文式 18-20）
        self.linear_l_at = nn.Linear(2 * d_k, d_k)  # 响应时间通道（W_i）
        self.linear_l_at_h = nn.Linear(2 * d_k, d_k)  # 响应时间通道（W_li）
        self.linear_l_it = nn.Linear(2 * d_k, d_k)  # 间隔时间通道（W_f）
        self.linear_l_it_h = nn.Linear(2 * d_k, d_k)  # 间隔时间通道（W_lf）
        for layer in (
            self.linear_l_at,
            self.linear_l_at_h,
            self.linear_l_it,
            self.linear_l_it_h,
        ):
            torch.nn.init.xavier_uniform_(layer.weight)

        # 预测层（论文式 22）
        self.linear_y = nn.Linear(d_e + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_y.weight)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        # 缓冲区
        self.q_gamma = q_gamma
        # 每题的贝叶斯难度（论文式 2）。
        self.register_buffer(
            "difficulty", torch.full((num_questions,), 0.5), persistent=False
        )
        # 全局交互特征字典（K-Means 中心），在线更新。
        self.register_buffer("global_dict", torch.zeros(global_dict_size, d_k))
        # LSAT 的因果掩码（上三角为 True = 屏蔽未来）。
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def set_difficulty(self, difficulty):
        """设置每题难度缓冲区。"""
        d = torch.as_tensor(difficulty, dtype=self.difficulty.dtype)
        self.difficulty.copy_(d.to(self.difficulty.device))

    def update_global_dict(self, centers: torch.Tensor):
        """用新的 K-Means 中心更新全局字典缓冲区。"""
        self.global_dict.copy_(centers.to(self.global_dict.device))

    # 交互嵌入
    def compute_interaction_embeddings(
        self,
        e_data: torch.Tensor,
        c_data: torch.Tensor,
        a_data: torch.Tensor,
    ) -> torch.Tensor:
        """计算原始交互嵌入 ``x_t``（CSAT 之前），用于 K-Means 聚类。

        Args:
            e_data: 题目序列 [B, S]
            c_data: 概念（主知识点）序列 [B, S]
            a_data: 答案序列 [B, S]

        Returns:
            [B, S, d_k] 的交互嵌入。
        """
        e_embed_data = self.e_embed(e_data)
        c_embed_data = self.c_embed(c_data)
        a_expand = a_data.unsqueeze(-1).expand(-1, -1, self.d_a)
        e_diff = self.difficulty[e_data]
        e_diff_expand = e_diff.unsqueeze(-1).expand(-1, -1, self.d_a)
        return self.linear_all_learning(
            torch.cat((e_embed_data, c_embed_data, a_expand, e_diff_expand), dim=2)
        )

    # 前向传播
    def forward(
        self,
        e_data: torch.Tensor,
        at_data: torch.Tensor,
        a_data: torch.Tensor,
        it_data: torch.Tensor,
        c_data: torch.Tensor,
        skills: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        约定：输出 ``pred[:, t]`` 为对第 t 个交互（``a_data[:, t]``）的预测，使用
        0..t-1 的信息（与原参考代码一致，t=0 处为零）。

        Args:
            e_data: 题目序列 [B, S]
            at_data: 响应时间桶序列 [B, S]
            a_data: 答案序列 [B, S]（同时作为输入特征与预测目标）
            it_data: 间隔时间桶序列 [B, S]
            c_data: 概念（主知识点）序列 [B, S]
            skills: 本次交互的完整技能列表 [B, S, K]，以 num_skills 填充

        Returns:
            pred: [B, S]，每个位置 ∈ (0, 1)。
        """
        bs, seq_len = e_data.size(0), e_data.size(1)
        d_k = self.d_k
        device = e_data.device

        e_embed_data = self.e_embed(e_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        c_embed_data = self.c_embed(c_data)

        # 答案 / 难度展开为 d_a 维向量
        a_expand = a_data.unsqueeze(-1).expand(-1, -1, self.d_a)
        e_diff = self.difficulty[e_data]
        e_diff_expand = e_diff.unsqueeze(-1).expand(-1, -1, self.d_a)

        # 初始每个概念的知识状态 h_0（xavier 初始化，论文式 17 之前）。
        h_pre = (
            nn.init.xavier_uniform_(torch.zeros(self.num_skills, d_k, device=device))
            .unsqueeze(0)
            .repeat(bs, 1, 1)
        )
        h_tilde_pre = None

        # 交互嵌入 x_t
        all_learning = self.linear_all_learning(
            torch.cat((e_embed_data, c_embed_data, a_expand, e_diff_expand), dim=2)
        )

        # 加位置编码，转置为 (seq, batch, d_k)
        pos_id = torch.arange(seq_len, device=device).unsqueeze(0)
        all_learning_att_in = all_learning + self.pos_embedding(pos_id)
        all_learning_att_in_t = all_learning_att_in.permute(1, 0, 2)

        # LSAT：因果自注意力（论文式 14）
        q_in = self.linear_is[2](all_learning_att_in_t)
        k_in = self.linear_is[1](all_learning_att_in_t)
        v_in = self.linear_is[0](all_learning_att_in_t)
        attn_out, _ = self.multi_attention_is(
            q_in, k_in, v_in, attn_mask=self.causal_mask, need_weights=False
        )
        attn_out = self.layer_norm1_is(attn_out + q_in)
        attn_out = attn_out.permute(1, 0, 2)  # -> (batch, seq, d_k)

        # GSAT：以全局字典为 K/V 的注意力（论文式 15）
        gd = self.global_dict.unsqueeze(1)  # [N, 1, d_k]
        k_in_hat = self.linear_cs[1](gd).expand(-1, bs, -1)  # [N, bs, d_k]
        v_in_hat = self.linear_cs[0](gd).expand(-1, bs, -1)
        q_in_hat = self.linear_cs[2](all_learning_att_in_t)  # [seq, bs, d_k]
        attn_out_hat, _ = self.multi_attention_cs(
            q_in_hat, k_in_hat, v_in_hat, need_weights=False
        )
        attn_out_hat = self.layer_norm1_cs(attn_out_hat + q_in_hat)
        attn_out_hat = attn_out_hat.permute(1, 0, 2)  # -> (batch, seq, d_k)

        # 融合 LSAT + GSAT -> 因果交互嵌入 x̂_t（论文式 16）
        all_learning = self.fc_att_cs_is(torch.cat([attn_out, attn_out_hat], dim=2))

        # LBS + 预测（逐时刻循环，论文式 17-22）
        pred = torch.zeros(bs, seq_len, device=device)
        skill_indicator = (
            skills_to_multihot(skills, self.num_skills).to(h_pre.dtype) + self.q_gamma
        )
        for t in range(seq_len - 1):
            q_e = skill_indicator[:, t].view(bs, 1, -1)
            at = at_embed_data[:, t]  # 响应时间嵌入 [bs, d_k]
            it = it_embed_data[:, t]  # 间隔时间嵌入 [bs, d_k]

            # 适配知识状态 ĥ_{t-1} = Q_C(q_t) · h_{t-1}（论文式 17，按概念求和）。
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(bs, d_k)

            learning = all_learning[:, t]  # 因果交互嵌入 x̂_t [bs, d_k]

            # 知识获得 + 输入门（使用响应时间 at），论文式 19-20。
            learning_at = self.linear_l_at(torch.cat((learning, at), dim=1))
            learning_at = self.linear_l_at_h(
                torch.cat((h_tilde_pre, learning_at), dim=1)
            )
            gamma_l = self.sig(learning_at)  # 输入门 η_t
            learning_gain = self.tanh(learning_at)  # 获得项 l_i_t
            l_input_tilde = gamma_l * ((learning_gain + 1) / 2)
            # 通过 q_e 把获得项分配到相关概念上。
            l_input = self.dropout(
                q_e.transpose(1, 2).bmm(l_input_tilde.view(bs, 1, d_k))
            )  # [bs, num_skills, d_k]

            # 遗忘门（使用间隔时间 it），论文式 18。
            learning_it = self.linear_l_it(torch.cat((learning, it), dim=1))
            gamma_f = self.sig(
                self.linear_l_it_h(torch.cat((h_tilde_pre, learning_it), dim=1))
            )  # [bs, d_k]

            # 更新知识状态 h_t = l_f_t · h_{t-1} + η_t · l_i_t（论文式 21）。
            h = h_pre * gamma_f.unsqueeze(1) + l_input

            # 预测下一题（论文式 22）。
            h_tilde = skill_indicator[:, t + 1].view(bs, 1, -1).bmm(h).view(bs, d_k)
            y = (
                self.sig(
                    self.linear_y(torch.cat((e_embed_data[:, t + 1], h_tilde), dim=1))
                ).sum(dim=1)
                / d_k
            )
            pred[:, t + 1] = y

            # 为下一时刻准备。
            h_pre = h
            h_tilde_pre = h_tilde

        return pred
