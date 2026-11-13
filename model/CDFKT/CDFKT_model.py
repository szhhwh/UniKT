"""Cognitive Decouple-then-Fuse Knowledge Tracing."""

import math

import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_q, num_c, emb_size, question_concept_map):
        super().__init__()
        self.pro_embed = nn.Parameter(torch.ones(num_q, emb_size))
        self.ans_embed = nn.Embedding(2, emb_size)
        concept_embedding = nn.Embedding(num_c, emb_size)
        nn.init.xavier_uniform_(concept_embedding.weight)

        valid = (question_concept_map != -1).unsqueeze(-1)
        concepts = F.embedding(
            question_concept_map.clamp(min=0), concept_embedding.weight
        )
        question_embedding = (concepts * valid.float()).sum(dim=1)
        question_embedding = question_embedding / valid.sum(dim=1).clamp(min=1.0)
        # The source trains the initialized question table, not the concept table.
        self.pro_embed = nn.Parameter(question_embedding * self.pro_embed)
        nn.init.xavier_uniform_(self.ans_embed.weight)

    def forward(self, last_pro, last_ans, next_pro):
        last_pro_embed = F.embedding(last_pro, self.pro_embed)
        next_pro_embed = F.embedding(next_pro, self.pro_embed)
        return last_pro_embed + self.ans_embed(last_ans), next_pro_embed


class AdaptiveFusion(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(3 * emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
            nn.Sigmoid(),
        )

    def forward(self, x, adaptability, accumulation):
        gate = self.gate_net(torch.cat([x, adaptability, accumulation], dim=-1))
        return gate * adaptability + (1 - gate) * accumulation, gate


class CDFKT(nn.Module):
    def __init__(
        self,
        num_q,
        num_c,
        question_concept_map,
        emb_size=512,
        dropout=0.1,
        base_intensity=0.1,
        alpha_init_scale=0.01,
        min_half_life=1.0,
        max_half_life=64.0,
        sa_layerscale_init=1.0,
        sa_dropout=0.0,
        sa_gate_bias=2.0,
        fusion_dropout=0.1,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.model_name = "cdfkt"
        self.embedding_layer = EmbeddingLayer(
            num_q, num_c, emb_size, question_concept_map
        )
        self.mu = nn.Parameter(torch.full((emb_size,), base_intensity))
        self.alpha = nn.Parameter(torch.randn(emb_size, emb_size) * alpha_init_scale)
        min_hl = max(float(min_half_life), 1e-3)
        max_hl = max(float(max_half_life), min_hl)
        half_lives = torch.logspace(math.log10(min_hl), math.log10(max_hl), emb_size)
        self.beta = nn.Parameter(math.log(2.0) / half_lives)

        self.sa_gate_proj = nn.Linear(emb_size, emb_size)
        nn.init.zeros_(self.sa_gate_proj.weight)
        nn.init.constant_(self.sa_gate_proj.bias, float(sa_gate_bias))
        self.sa_layernorm = nn.LayerNorm(emb_size)
        self.sa_layerscale = nn.Parameter(
            torch.full((emb_size,), float(sa_layerscale_init))
        )
        self.sa_drop = nn.Dropout(sa_dropout) if sa_dropout > 0 else nn.Identity()

        self.knowledge_accumulation_encoder = Mamba(
            d_model=emb_size, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.fusion_layer = AdaptiveFusion(emb_size, fusion_dropout)
        self.out_layer = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, 1),
        )

    def _excitation_input(self, xemb):
        contributions = torch.matmul(xemb, self.alpha.transpose(0, 1))
        return contributions * torch.sigmoid(self.sa_gate_proj(xemb))

    def _causal_decay_conv(self, contributions, kernel):
        _, seq_len, emb_size = contributions.shape
        x = F.pad(contributions.transpose(1, 2), (seq_len - 1, 0))
        decayed = F.conv1d(x, kernel.flip(-1).unsqueeze(1), groups=emb_size)
        return decayed.transpose(1, 2)

    def _decay_kernel(self, seq_len, dtype, device):
        time_indices = torch.arange(seq_len, device=device, dtype=dtype)
        beta = self.beta.clamp(min=1e-6)
        return torch.exp(-beta.unsqueeze(1) * time_indices.unsqueeze(0))

    def situational_adaptability_encoder(self, xemb):
        contributions = self._excitation_input(xemb)
        kernel = self._decay_kernel(xemb.shape[1], xemb.dtype, xemb.device)
        decayed = self.sa_layernorm(self._causal_decay_conv(contributions, kernel))
        return self.sa_drop(decayed) * self.sa_layerscale

    def forward(self, question, response):
        """Return [B, S-1] probabilities for the next question."""
        xemb, next_xemb = self.embedding_layer(
            question[:, :-1], response[:, :-1], question[:, 1:]
        )
        temporal = self.situational_adaptability_encoder(xemb)
        adaptability = xemb + self.mu.unsqueeze(0).unsqueeze(0) + temporal
        accumulation = self.knowledge_accumulation_encoder(xemb)
        fused, _ = self.fusion_layer(xemb, adaptability, accumulation)
        return torch.sigmoid(
            self.out_layer(torch.cat([fused, next_xemb], dim=-1))
        ).squeeze(-1)
