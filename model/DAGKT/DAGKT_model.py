from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import Linear

from ..layers import GNN_QS, GeneralInteraction, HistoryRecap


class Autoencoder(nn.Module):
    """自编码器模块，用于难度和尝试次数的特征编码。

    架构: 1 → ae_hidden_dim → embedding_dim → ae_hidden_dim → 1
    编码器和解码器均使用 Sigmoid 激活函数。

    Args:
        input_dim: 输入维度（默认 1，对应标量输入）
        ae_hidden_dim: 自编码器隐藏层维度
        embedding_dim: 编码后的输出维度
    """

    def __init__(
        self,
        input_dim: int = 1,
        ae_hidden_dim: int = 50,
        embedding_dim: int = 100,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ae_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(ae_hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, ae_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(ae_hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入张量 [..., input_dim]

        Returns:
            (encoded, reconstruction_loss) 元组:
                - encoded: 编码后的表示 [..., embedding_dim]
                - reconstruction_loss: 标量 MSE 重建损失
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        recon_loss = F.mse_loss(decoded, x)
        return encoded, recon_loss


class DAGKT(nn.Module):
    """DAGKT 主模型。

    在 GIKT 基础上增加难度模块和尝试次数模块：
    - 难度模块: 用自编码器编码题目正确率，与题目嵌入融合
    - 尝试次数模块: 用自编码器编码学生对每题的尝试次数，与答案嵌入融合

    总损失 = 预测损失(BCE) + 难度自编码器重建损失 + 尝试次数自编码器重建损失

    Args:
        data_metadata: 数据集元数据，包含 num_questions 和 num_skills
        question_difficulty: 题目正确率张量 [num_questions, 1]
        embedding_dim: Embedding 维度
        hidden_dim: 隐藏层维度
        lstm_layers: LSTM 层数
        dropout: Dropout 率
        ae_hidden_dim: 自编码器隐藏层维度
        n_hop: GNN 跳数
        heads: 注意力头数
        history_neighbour: 历史邻居数量
        att_bound: 注意力边界
        **kwargs: 额外的关键字参数
    """

    def __init__(
        self,
        data_metadata: dict[str, Any],
        question_difficulty: torch.Tensor | None = None,
        *,
        embedding_dim: int = 100,
        hidden_dim: int = 100,
        lstm_layers: int = 2,
        dropout: float = 0.4,
        ae_hidden_dim: int = 50,
        n_hop: int = 3,
        heads: int = 2,
        history_neighbour: int = 5,
        att_bound: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_metadata = data_metadata

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        num_questions = data_metadata["num_questions"]
        num_skills = data_metadata["num_skills"]

        # Question correct rates: registered as a buffer (tracked, not trained)
        if question_difficulty is not None:
            self.register_buffer(
                "difficulty_rates", question_difficulty.float(), persistent=False
            )
        else:
            self.register_buffer(
                "difficulty_rates", torch.zeros(num_questions, 1), persistent=False
            )

        self.question_embedding = nn.Embedding(
            num_embeddings=num_questions,
            embedding_dim=self.embedding_dim,
        )
        self.skill_embedding = nn.Embedding(
            num_embeddings=num_skills,
            embedding_dim=self.embedding_dim,
        )
        self.answer_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        self.conv = GNN_QS(
            embedding_dim=self.embedding_dim,
            n_hop=n_hop,
            heads=heads,
            dropout=self.dropout,
        )

        self.difficulty_ae = Autoencoder(
            input_dim=1,
            ae_hidden_dim=ae_hidden_dim,
            embedding_dim=self.embedding_dim,
        )

        self.attempt_ae = Autoencoder(
            input_dim=1,
            ae_hidden_dim=ae_hidden_dim,
            embedding_dim=self.embedding_dim,
        )

        self.transdiff = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

        self.trans_answer = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

        self.fc_exercise = Linear(
            2 * self.embedding_dim, self.hidden_dim, weight_initializer="uniform"
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.history_review = HistoryRecap(
            hist_neighbor_num=history_neighbour,
            att_bound=att_bound,
        )

        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

    def forward(
        self,
        user_sequence: torch.Tensor,  # [B, S]
        user_response: torch.Tensor,  # [B, S]
        user_mask: torch.Tensor,  # [B, S]
        graph: Any,
        skills: torch.Tensor,  # [B, S, K]
        attempt_counts: torch.Tensor,  # [B, S]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            user_sequence: 用户问题序列 [B, S]
            user_response: 用户回答序列 [B, S]
            user_mask: 有效位置掩码 [B, S]
            graph: 问题-技能异构图
            skills: 本次交互的技能列表 [B, S, K]，以 num_skills 填充
            attempt_counts: 每个学生对每题的累计尝试次数 [B, S]

        Returns:
            (logits, loss_diff, loss_attempt) 元组
        """
        B, S = user_sequence.size()

        diff_emb, loss_diff = self.difficulty_ae(self.difficulty_rates)

        conv = self.conv(
            {
                "question": self.question_embedding.weight,
                "skill": self.skill_embedding.weight,
            },
            graph.edge_index_dict,
        )
        question_conv: torch.Tensor = conv["question"]  # [num_questions, E]
        skill_conv: torch.Tensor = conv["skill"]  # [num_skills, E]

        q_emb = question_conv[user_sequence]  # [B, S, E]
        cur_diff_emb = diff_emb[user_sequence]  # [B, S, E]
        q_fused = torch.relu(
            self.transdiff(torch.cat([q_emb, cur_diff_emb], dim=-1))
        )  # [B, S, E]

        attempt_input = attempt_counts.reshape(-1, 1)  # [B*S, 1]
        attempt_encoded, loss_attempt = self.attempt_ae(attempt_input)
        # [B*S, E] → [B, S, E]
        attempt_emb = attempt_encoded.reshape(B, S, self.embedding_dim)

        ans_emb = self.answer_embedding(user_response)  # [B, S, E]
        ans_fused = torch.relu(
            self.trans_answer(torch.cat([ans_emb, attempt_emb], dim=-1))
        )  # [B, S, E]

        exercise_emb = torch.cat([q_fused, ans_fused], dim=-1)  # [B, S, 2E]
        exercise_emb = torch.relu(self.fc_exercise(exercise_emb))  # [B, S, H]
        exercise_emb = self.embedding_dropout(exercise_emb)

        lstm_output, _ = self.lstm(exercise_emb)  # [B, S, H]

        next_user_sequence = torch.zeros_like(user_sequence)  # [B, S]
        if S > 1:
            next_user_sequence[:, :-1] = user_sequence[:, 1:]
            next_user_sequence[:, -1] = 0

        next_q_emb = question_conv[next_user_sequence]  # [B, S, E]
        next_diff_emb = diff_emb[next_user_sequence]  # [B, S, E]
        next_q_fused = torch.relu(
            self.transdiff(torch.cat([next_q_emb, next_diff_emb], dim=-1))
        )  # [B, S, E]

        history_question_neighbors = self.history_review(
            q_emb,
            next_q_emb,
            exercise_emb,
            user_mask,
        )  # [B, S, M, H]

        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )  # [B, S, M+1, H]

        related_skill_ids = torch.cat([skills[:, 1:], skills[:, -1:]], dim=1)
        skill_slot_mask = related_skill_ids != skill_conv.size(0)
        skill_conv_padded = F.pad(skill_conv, (0, 0, 0, 1))
        related_skill_embs = skill_conv_padded[related_skill_ids]

        # knowledge_status uses next_q_fused (already difficulty-fused) as the question representation
        knowledge_status = torch.cat(
            [next_q_fused.unsqueeze(2), related_skill_embs],
            dim=2,
        )  # [B, S, K+1, E]

        logits = self.general_interaction(
            student_status,
            knowledge_status,
            user_mask,
            skill_slot_mask=skill_slot_mask,
        )  # [B, S]

        return logits, loss_diff, loss_attempt
