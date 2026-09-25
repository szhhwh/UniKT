"""DJIM dual-state module composed with the existing DKT and SAINT models."""

from __future__ import annotations

import math

import torch
from torch import nn

from model.DJIM.knowledge_adapter import knowledge_prediction
from model.DKT.DKT_model import DKT
from model.SAINT.SAINT_model import SAINT


class BehaviorAttentionLayer(nn.Module):
    """Paper Eqs. (4)-(8): exercise query over per-response behavior features."""

    def __init__(self, dim: int = 128, heads: int = 3, head_dim: int = 128):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.query = nn.Linear(dim, heads * head_dim, bias=False)
        self.key = nn.Linear(dim, heads * head_dim, bias=False)
        self.value = nn.Linear(dim, heads * head_dim, bias=False)
        self.residual = nn.Linear(dim, heads * head_dim, bias=False)
        self.next_query = nn.Linear(heads * head_dim, dim)

    def forward(
        self,
        exercise: torch.Tensor,
        features: torch.Tensor,
        original_exercise: torch.Tensor,
    ):
        batch, length, count, _ = features.shape
        query = self.query(exercise).reshape(batch, length, self.heads, self.head_dim)
        key = self.key(features).reshape(
            batch, length, count, self.heads, self.head_dim
        )
        value = self.value(features).reshape(
            batch, length, count, self.heads, self.head_dim
        )
        scores = (query.unsqueeze(2) * key).sum(-1) / math.sqrt(self.head_dim)
        weights = scores.softmax(dim=2)
        relation = (weights.unsqueeze(-1) * value).sum(2).flatten(-2)
        combined = torch.relu(relation + self.residual(original_exercise))
        return combined, self.next_query(combined)


class DJIM(nn.Module):
    """Causal DJIM-DKT or DJIM-SAINT.

    The operators in paper Eqs. (2)-(13) are retained. Predictions use the
    current exercise and only earlier responses/behavior, because the paper's
    same-position notation otherwise exposes the target response.
    """

    def __init__(
        self,
        base_model: str,
        num_questions: int,
        num_skills: int,
        seq_len: int,
        num_features: int = 3,
        dim: int = 128,
        psychology_heads: int = 3,
        psychology_layers: int = 2,
        dropout: float = 0.2,
        saint_heads: int = 8,
        saint_blocks: int = 2,
        q_learning_rate: float = 0.1,
        q_discount: float = 0.9,
        q_epsilon: float = 0.1,
    ):
        super().__init__()
        if base_model not in {"DKT", "SAINT"}:
            raise ValueError("base_model must be DKT or SAINT")
        if psychology_layers < 2:
            raise ValueError("the paper stacks multiple psychology attention layers")
        self.base_model_name = base_model
        self.num_features = num_features
        self.dim = dim
        self.q_learning_rate = q_learning_rate
        self.q_discount = q_discount
        self.q_epsilon = q_epsilon

        self.question_embedding = nn.Embedding(num_questions, dim)
        self.skill_embedding = nn.Embedding(num_skills, dim)
        self.exercise_projection = nn.Linear(2 * dim, dim)
        self.correct_projection = nn.Linear(dim, dim)
        self.incorrect_projection = nn.Linear(dim, dim)

        self.behavior_projections = nn.ModuleList(
            nn.Linear(1, dim) for _ in range(num_features)
        )
        self.psychology_layers = nn.ModuleList(
            BehaviorAttentionLayer(dim, psychology_heads, dim)
            for _ in range(psychology_layers)
        )
        self.psychology_output = nn.Linear(psychology_heads * dim, 1)

        if base_model == "DKT":
            self.knowledge_model = DKT(num_skills, dim, dropout)
        else:
            self.knowledge_model = SAINT(
                num_questions,
                num_skills,
                seq_len,
                dim,
                saint_heads,
                dropout,
                saint_blocks,
            )

        # Quantized tabular Q(s,a). The paper does not define discretization
        # of its continuous state; this is the explicit tabular realization.
        self.register_buffer("q_table", torch.zeros(11 * 11 * 2 * 11, 11))

    def _q_states(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        previous_response: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        a = (alpha.detach() * 10).round().long().clamp(0, 10)
        b = (beta.detach() * 10).round().long().clamp(0, 10)
        return ((a * 11 + b) * 2 + previous_response) * 11 + previous_action

    @staticmethod
    def joint_probability(
        alpha: torch.Tensor, beta: torch.Tensor, lambda_k: torch.Tensor
    ) -> torch.Tensor:
        """Paper Eqs. (10)-(13), with lambda_p = 1 - lambda_k."""
        lambda_p = 1 - lambda_k
        mu_pk = torch.where(
            beta >= 0.5,
            1 + beta / (beta + lambda_k * (1 - beta)).clamp_min(1e-8),
            beta / (beta + lambda_p * (1 - beta)).clamp_min(1e-8),
        )
        mu_kp = torch.where(
            alpha >= 0.5,
            1 + alpha / (alpha + lambda_p * (1 - alpha)).clamp_min(1e-8),
            alpha / (alpha + lambda_k * (1 - alpha)).clamp_min(1e-8),
        )
        return torch.tanh(lambda_k * mu_pk * alpha + lambda_p * mu_kp * beta)

    @torch.no_grad()
    def _update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor | None,
        valid: torch.Tensor,
        next_valid: torch.Tensor | None = None,
    ) -> None:
        if not valid.any():
            return
        states, actions, rewards = states[valid], actions[valid], rewards[valid]
        future = (
            self.q_table[next_states[valid]].max(-1).values
            if next_states is not None
            else torch.zeros_like(rewards)
        )
        if next_valid is not None:
            future = future * next_valid[valid]
        target = rewards + self.q_discount * future
        flat = states * 11 + actions
        old = self.q_table.view(-1)
        residual = target - old[flat]
        sums = torch.zeros_like(old)
        counts = torch.zeros_like(old)
        sums.index_add_(0, flat, residual)
        counts.index_add_(0, flat, torch.ones_like(residual))
        touched = counts > 0
        old[touched] += self.q_learning_rate * sums[touched] / counts[touched]

    def forward(
        self,
        question: torch.Tensor,
        skill: torch.Tensor,
        response: torch.Tensor,
        behavior: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if question.shape != skill.shape or question.shape != response.shape:
            raise ValueError("question, skill and response shapes must match")
        if behavior.shape != (*question.shape, self.num_features):
            raise ValueError("behavior must have shape [B, S, num_features]")
        if valid is None:
            valid = torch.ones_like(response, dtype=torch.bool)

        exercise = self.exercise_projection(
            torch.cat(
                (self.question_embedding(question), self.skill_embedding(skill)), -1
            )
        )
        interaction = torch.where(
            response.unsqueeze(-1).bool(),
            self.correct_projection(exercise),
            self.incorrect_projection(exercise),
        )
        # Knowledge estimates at t use interactions through t-1.
        alpha = knowledge_prediction(
            self.knowledge_model, question, skill, response, valid, interaction
        )
        if self.base_model_name == "DKT":
            alpha = torch.cat((torch.full_like(alpha[:, :1], 0.5), alpha[:, 1:]), 1)

        history_behavior = torch.cat(
            (torch.zeros_like(behavior[:, :1]), behavior[:, :-1]), dim=1
        )
        encoded = torch.stack(
            [
                layer(history_behavior[..., i : i + 1])
                for i, layer in enumerate(self.behavior_projections)
            ],
            dim=2,
        )
        query = exercise
        for layer in self.psychology_layers:
            combined, query = layer(query, encoded, exercise)
        beta = torch.sigmoid(self.psychology_output(combined)).squeeze(-1)

        predictions = []
        previous_action = torch.full_like(response[:, 0], 5)
        previous_response = torch.zeros_like(response[:, 0])
        last_state = last_action = last_reward = last_valid = None
        for t in range(response.shape[1]):
            state = self._q_states(
                alpha[:, t], beta[:, t], previous_response, previous_action
            )
            if self.training and last_state is not None:
                self._update_q(
                    last_state,
                    last_action,
                    last_reward,
                    state,
                    last_valid,
                    valid[:, t],
                )
            action = self.q_table[state].argmax(-1)
            if self.training and self.q_epsilon:
                explore = torch.rand_like(alpha[:, t]) < self.q_epsilon
                action = torch.where(
                    explore,
                    torch.randint(11, action.shape, device=action.device),
                    action,
                )
            lambda_k = action.to(alpha.dtype) / 10
            prediction = self.joint_probability(alpha[:, t], beta[:, t], lambda_k)
            predictions.append(prediction)
            last_state, last_action = state, action
            last_reward = (1 - (response[:, t] - prediction.detach()).abs()).detach()
            last_valid = valid[:, t]
            previous_response = response[:, t]
            previous_action = action
        if self.training and last_state is not None:
            self._update_q(last_state, last_action, last_reward, None, last_valid)
        return torch.stack(predictions, dim=1)
