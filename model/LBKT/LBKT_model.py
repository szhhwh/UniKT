import torch
import torch.nn as nn

from model.layers.interaction import skills_to_multihot


class Layer1(nn.Module):
    """行为效应门控层。"""

    def __init__(self, num_units: int, d: float = 10, k: float = 0.3, b: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(2 * num_units, num_units))
        self.bias = nn.Parameter(torch.zeros(1, num_units))
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.bias)
        self.d = d
        self.k = k
        self.b = b

    def forward(
        self, factor: torch.Tensor, interact_emb: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        gate = self.k + (1 - self.k) / (1 + torch.exp(-self.d * (factor - self.b)))
        w = torch.cat([h, interact_emb], -1).matmul(self.weight) + self.bias
        return torch.sigmoid(w * gate)


class LBKTcell(nn.Module):
    """LBKT 核心循环单元。"""

    def __init__(
        self, num_units: int, memory_size: int, dim_tp: int, dropout: float = 0.2
    ):
        super().__init__()
        self.num_units = num_units
        self.memory_size = memory_size
        self.factor_dim = 50
        self.r = 4

        self.time_gain = Layer1(num_units)
        self.attempt_gain = Layer1(num_units)
        self.hint_gain = Layer1(num_units)

        self.time_weight = nn.Parameter(torch.Tensor(self.r, num_units + 1, num_units))
        self.attempt_weight = nn.Parameter(
            torch.Tensor(self.r, num_units + 1, num_units)
        )
        self.hint_weight = nn.Parameter(torch.Tensor(self.r, num_units + 1, num_units))
        nn.init.xavier_normal_(self.time_weight)
        nn.init.xavier_normal_(self.attempt_weight)
        nn.init.xavier_normal_(self.hint_weight)

        self.Wf = nn.Parameter(torch.Tensor(1, self.r))
        nn.init.xavier_normal_(self.Wf)

        self.bias = nn.Parameter(torch.Tensor(1, num_units))
        nn.init.xavier_normal_(self.bias)

        # Forget gate: a single Linear is decomposed into per-input-segment weights
        # Inputs: h_pre(num_units), interact_emb(num_units), time(factor_dim), attempt(factor_dim), hint(factor_dim)
        self.gate_w_h = nn.Parameter(torch.Tensor(num_units, num_units))
        self.gate_w_interact = nn.Parameter(torch.Tensor(num_units, num_units))
        self.gate_w_time = nn.Parameter(torch.Tensor(num_units, self.factor_dim))
        self.gate_w_attempt = nn.Parameter(torch.Tensor(num_units, self.factor_dim))
        self.gate_w_hint = nn.Parameter(torch.Tensor(num_units, self.factor_dim))
        self.gate_bias = nn.Parameter(torch.Tensor(num_units))
        nn.init.xavier_normal_(self.gate_w_h)
        nn.init.xavier_normal_(self.gate_w_interact)
        nn.init.xavier_normal_(self.gate_w_time)
        nn.init.xavier_normal_(self.gate_w_attempt)
        nn.init.xavier_normal_(self.gate_w_hint)
        nn.init.zeros_(self.gate_bias)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dim_tp + num_units, num_units)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(
        self,
        interact_emb: torch.Tensor,
        correlation_weight: torch.Tensor,
        topic_emb: torch.Tensor,
        time_factor: torch.Tensor,
        attempt_factor: torch.Tensor,
        hint_factor: torch.Tensor,
        h_pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Weighted knowledge-state extraction -> [B, num_units]
        h_pre_tilde = torch.einsum("bm,bmn->bn", correlation_weight, h_pre)

        preds = (
            torch.sum(
                torch.sigmoid(
                    self.output_layer(torch.cat([h_pre_tilde, topic_emb], -1))
                ),
                -1,
            )
            / self.num_units
        )

        time_gain = self.time_gain(time_factor, interact_emb, h_pre_tilde)
        attempt_gain = self.attempt_gain(attempt_factor, interact_emb, h_pre_tilde)
        hint_gain = self.hint_gain(hint_factor, interact_emb, h_pre_tilde)

        pad = torch.ones_like(time_factor)
        fusion_all = (
            torch.matmul(torch.cat([time_gain, pad], -1), self.time_weight)
            * torch.matmul(torch.cat([attempt_gain, pad], -1), self.attempt_weight)
            * torch.matmul(torch.cat([hint_gain, pad], -1), self.hint_weight)
        )
        fusion_all = (
            torch.matmul(self.Wf, fusion_all.permute(1, 0, 2)).squeeze(1) + self.bias
        )
        learning_gain = torch.relu(fusion_all)

        LG = torch.einsum("bm,bn->bmn", correlation_weight, learning_gain)

        # h_pre: [B, M, num_units] -> matmul -> [B, M, num_units]
        # other inputs are [B, dim], broadcast-added to [B, M, num_units]
        forget_gate = (
            torch.matmul(h_pre, self.gate_w_h.T)
            + torch.matmul(interact_emb, self.gate_w_interact.T).unsqueeze(1)
            + (time_factor * self.gate_w_time.sum(dim=1)).unsqueeze(1)
            + (attempt_factor * self.gate_w_attempt.sum(dim=1)).unsqueeze(1)
            + (hint_factor * self.gate_w_hint.sum(dim=1)).unsqueeze(1)
            + self.gate_bias
        )

        LG = self.dropout(LG)
        h = h_pre * torch.sigmoid(forget_gate) + LG

        return preds, h


class LBKT(nn.Module):
    def __init__(
        self,
        dim_tp: int,
        dim_hidden: int,
        num_units: int,
        dropout: float,
        data_metadata: dict,
        q_gamma: float = 0.1,
    ):
        super().__init__()
        num_questions = data_metadata["num_questions"]
        num_skills = data_metadata["num_skills"]
        memory_size = num_skills

        self.memory_size = memory_size
        self.q_gamma = q_gamma
        self.num_units = num_units

        self.embedding_topic = nn.Embedding(num_questions, dim_tp)
        nn.init.xavier_normal_(self.embedding_topic.weight)

        self.embedding_resps = nn.Embedding(2, dim_hidden)
        nn.init.xavier_normal_(self.embedding_resps.weight)

        self.input_layer = nn.Linear(dim_tp + dim_hidden, num_units)
        nn.init.xavier_normal_(self.input_layer.weight)

        self.lbkt_cell = LBKTcell(num_units, memory_size, dim_tp, dropout=dropout)

        self.init_h = nn.Parameter(torch.Tensor(memory_size, num_units))
        nn.init.xavier_normal_(self.init_h)

    def forward(
        self,
        topics: torch.Tensor,
        resps: torch.Tensor,
        time_factor: torch.Tensor,
        attempt_factor: torch.Tensor,
        hint_factor: torch.Tensor,
        skills: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = topics.size()

        topic_emb = self.embedding_topic(topics)
        resps_emb = self.embedding_resps(resps)
        skill_indicator = skills_to_multihot(skills, self.memory_size).to(
            topic_emb.dtype
        )
        correlation_weight = (1 - self.q_gamma) * skill_indicator + self.q_gamma

        acts_emb = torch.relu(self.input_layer(torch.cat([topic_emb, resps_emb], -1)))

        time_factor = time_factor.unsqueeze(-1)
        attempt_factor = attempt_factor.unsqueeze(-1)
        hint_factor = hint_factor.unsqueeze(-1)

        h_pre = self.init_h.unsqueeze(0).expand(batch_size, -1, -1)
        preds = torch.zeros(batch_size, seq_len, device=topics.device)

        for t in range(seq_len):
            pred, h = self.lbkt_cell(
                acts_emb[:, t],
                correlation_weight[:, t],
                topic_emb[:, t],
                time_factor[:, t],
                attempt_factor[:, t],
                hint_factor[:, t],
                h_pre,
            )
            h_pre = h
            preds[:, t] = pred

        return preds
