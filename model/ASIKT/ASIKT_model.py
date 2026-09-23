"""ASIKT difficulty-aware attention and selective state-space model."""

import math

import torch
from mamba_ssm import Mamba2
from torch import nn
from torch.nn import functional as F


def local_accuracy(ids, answers, valid, prior, frequency):
    """ASIKT's per-position accuracy: previous answer credited to current id."""
    batch, length = ids.shape
    positions = torch.arange(length, device=ids.device)
    active = valid & (positions[None, :] > 0)
    same_id = ids[:, :, None] == ids[:, None, :]
    history = positions[None, None, :] <= positions[None, :, None]
    selected = same_id & history & active[:, None, :]
    counts = selected.sum(-1)
    previous = F.pad(answers[:, :-1].float(), (1, 0))
    correct = (selected * previous[:, None, :]).sum(-1)
    baseline = prior[ids]
    base_frequency = frequency[ids]
    estimate = (correct + base_frequency * baseline) / (
        counts + base_frequency
    ).clamp_min(1)
    return torch.where(active, estimate, baseline)


def exercise_weights(
    question, skill, answer, valid, q_prior, q_freq, s_prior, s_freq, theta
):
    q_accuracy = local_accuracy(question, answer, valid, q_prior, q_freq)
    s_accuracy = local_accuracy(skill, answer, valid, s_prior, s_freq)
    difficulty = (1 - q_accuracy) * valid
    base = difficulty[:, :, None] * difficulty[:, None, :]
    same_skill = skill[:, :, None] == skill[:, None, :]
    skill_factor = (1 - s_accuracy).exp()[:, None, :]
    q_weight = base * torch.where(same_skill, skill_factor, 1.0)
    response_factor = torch.where(answer[:, :, None] == answer[:, None, :], 1.0, theta)
    qa_weight = base * torch.where(same_skill, skill_factor, response_factor)
    return q_weight, qa_weight


class DifficultyAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Parameter(torch.empty(heads, 1, 1))
        self.alpha = nn.Parameter(torch.empty(heads, 1, 1))
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.alpha)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.v_linear.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, weight, valid):
        batch, length, dim = x.shape
        # Original ASIKT uses the key projection for queries too.
        q = (
            self.k_linear(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        k = q
        v = (
            self.v_linear(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        positions = torch.arange(length, device=x.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        decay = torch.sigmoid(self.a) + (1 - torch.sigmoid(self.a)) * torch.exp(
            -F.softplus(self.alpha) * distance
        )
        scores = scores * weight[:, None] * decay[None]
        causal = positions[:, None] >= positions[None, :]
        allowed = causal[None, None] & valid[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention = self.dropout(scores.softmax(-1))
        output = attention @ v
        output = output.transpose(1, 2).contiguous().reshape(batch, length, dim)
        return self.out_proj(output) * valid[:, :, None]


class TransformerLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attention = DifficultyAttention(d_model, heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, weight, valid):
        x = self.norm1(x + self.dropout1(self.attention(x, weight, valid)))
        return self.norm2(x + self.dropout2(self.ff(x)))


class ASIKT(nn.Module):
    def __init__(
        self,
        n_skill,
        n_question,
        skill_prior,
        skill_frequency,
        question_prior,
        question_frequency,
        d_model=128,
        dropout=0.2,
        n_blocks=2,
        rasch=True,
        final_fc_dim=512,
        n_heads=8,
        d_ff=1024,
        theta=0.5,
        d_state=128,
        chunk_size=256,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.register_buffer("skill_prior", F.pad(skill_prior, (1, 0)))
        self.register_buffer("skill_frequency", F.pad(skill_frequency, (1, 0)))
        self.register_buffer("question_prior", F.pad(question_prior, (1, 0)))
        self.register_buffer("question_frequency", F.pad(question_frequency, (1, 0)))
        self.theta = theta
        self.rasch = rasch
        self.q_embed = nn.Embedding(n_question + 1, d_model, padding_idx=0)
        self.kc_embed = nn.Embedding(n_skill + 1, d_model, padding_idx=0)
        self.a_embed = nn.Embedding(2, d_model)
        if rasch:
            self.deviates_para = nn.Embedding(n_question + 1, 1, padding_idx=0)
            self.kc_embed_deviates = nn.Embedding(n_skill + 1, d_model, padding_idx=0)
            self.a_embed_deviates = nn.Embedding(2, d_model)
            nn.init.zeros_(self.deviates_para.weight)
        self.encoders = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerLayer(d_model, n_heads, d_ff, dropout)
                        for _ in range(n_blocks)
                    ]
                )
                for _ in range(4)
            ]
        )
        self.retriever = Mamba2(d_model * 4, d_state=d_state, chunk_size=chunk_size)
        self.linear = nn.Linear(d_model * 4, d_model)
        self.refiner = Mamba2(d_model, d_state=d_state, chunk_size=chunk_size)

        def head():
            return nn.Sequential(
                nn.Linear(d_model * 2, final_fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(final_fc_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        self.out = head()
        self.slipping = head()
        self.guessing = head()

    def forward(self, question, skill, answer, valid):
        valid = valid.bool()
        question = question.masked_fill(~valid, 0) + valid.long()
        skill = skill.masked_fill(~valid, 0) + valid.long()
        answer = answer.masked_fill(~valid, 0).long()
        q_weight, qa_weight = exercise_weights(
            question,
            skill,
            answer,
            valid,
            self.question_prior,
            self.question_frequency,
            self.skill_prior,
            self.skill_frequency,
            self.theta,
        )
        q = self.q_embed(question)
        kc = self.kc_embed(skill)
        a = self.a_embed(answer)
        qa = q + a
        kca = kc + a
        if self.rasch:
            deviation = self.deviates_para(question)
            kc_delta = self.kc_embed_deviates(skill)
            q = kc + deviation * kc_delta
            qa = kca + deviation * (self.a_embed_deviates(answer) + kc_delta)
        streams = [q, kc, qa, kca]
        weights = [q_weight, q_weight, qa_weight, qa_weight]
        encoded = []
        for stream, blocks, weight in zip(streams, self.encoders, weights, strict=True):
            for block in blocks:
                stream = block(stream, weight, valid)
            encoded.append(stream)
        q, kc, qa, kca = encoded
        future_q = F.pad(q[:, 1:], (0, 0, 0, 1))
        future_kc = F.pad(kc[:, 1:], (0, 0, 0, 1))
        x = torch.cat((qa, kca, future_q, future_kc), dim=-1)
        h = torch.sigmoid(self.linear(self.retriever(x)))
        y = self.refiner(h)
        readout = torch.cat((y, self.kc_embed(skill)), dim=-1)
        state = self.out(readout)
        guess = self.guessing(readout)
        slip = self.slipping(readout)
        logits = (1 - state) * guess + state * (1 - slip)
        return torch.sigmoid(logits.squeeze(-1))
