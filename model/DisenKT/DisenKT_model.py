"""DisenKT equations (1)--(18), with domain-local branches."""

import copy

import torch
import torch.nn.functional as F
from torch import nn


class ConceptGCN(nn.Module):
    """Residual symmetric-normalized concept propagation, Eq. (2)."""

    def __init__(self, hidden_size: int, layers: int):
        super().__init__()
        self.weights = nn.ParameterList(
            nn.Parameter(torch.empty(hidden_size, hidden_size)) for _ in range(layers)
        )
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for weight in self.weights:
            features = F.relu(torch.sparse.mm(adjacency, features @ weight) + features)
        return features


class VariationalAttention(nn.Module):
    """Interaction self-attention then historical question-to-interaction attention."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.interaction_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.interaction_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.mean = nn.Linear(hidden_size, hidden_size)
        self.variance = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, questions: torch.Tensor, interactions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = questions.size(1)
        positions = torch.arange(seq_len, device=questions.device)
        causal = positions[None, :] > positions[:, None]
        enriched, _ = self.interaction_attention(
            interactions,
            interactions,
            interactions,
            attn_mask=causal,
            key_padding_mask=~mask,
            need_weights=False,
        )
        enriched = self.interaction_norm(interactions + self.dropout(enriched))

        context = torch.zeros_like(questions)
        if seq_len > 1:
            # Query t sees interaction positions 0..t-1; t=0 has no history.
            history_mask = positions[None, :] >= positions[1:, None]
            previous, _ = self.cross_attention(
                questions[:, 1:],
                enriched,
                enriched,
                attn_mask=history_mask,
                key_padding_mask=~mask,
                need_weights=False,
            )
            context = torch.cat((context[:, :1], previous), dim=1)

        state = questions + context
        mean = F.relu(self.mean(state))
        variance = F.relu(self.variance(state)).clamp_min(1e-6)
        return mean, variance


class DomainBranch(nn.Module):
    def __init__(
        self,
        num_skills: int,
        max_seq_len: int,
        hidden_size: int,
        num_heads: int,
        gcn_layers: int,
        dropout: float,
        adjacency: torch.Tensor,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.question_shared = nn.Embedding(num_skills + 1, hidden_size, padding_idx=0)
        self.question_exclusive = nn.Embedding(
            num_skills + 1, hidden_size, padding_idx=0
        )
        self.interaction_shared = nn.Embedding(
            2 * num_skills + 1, hidden_size, padding_idx=0
        )
        self.interaction_exclusive = nn.Embedding(
            2 * num_skills + 1, hidden_size, padding_idx=0
        )
        self.position_shared = nn.Embedding(max_seq_len, hidden_size)
        self.position_exclusive = nn.Embedding(max_seq_len, hidden_size)
        self.gcn_shared = ConceptGCN(hidden_size, gcn_layers)
        self.gcn_exclusive = ConceptGCN(hidden_size, gcn_layers)
        self.question_norm_shared = nn.LayerNorm(hidden_size)
        self.question_norm_exclusive = nn.LayerNorm(hidden_size)
        self.encoder_shared = VariationalAttention(hidden_size, num_heads, dropout)
        self.encoder_exclusive = VariationalAttention(hidden_size, num_heads, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.register_buffer("adjacency", adjacency)

    def _sample(self, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mean
        return mean + variance.sqrt() * torch.randn_like(mean)

    def _question_embeddings(self, skills: torch.Tensor):
        seq_len = skills.size(1)
        positions = torch.arange(seq_len, device=skills.device)
        shared_graph = self.gcn_shared(self.question_shared.weight, self.adjacency)
        exclusive_graph = self.gcn_exclusive(
            self.question_exclusive.weight, self.adjacency
        )
        shared = self.question_norm_shared(
            shared_graph[skills]
            + self.question_shared(skills)
            + self.position_shared(positions)
        )
        exclusive = self.question_norm_exclusive(
            exclusive_graph[skills]
            + self.question_exclusive(skills)
            + self.position_exclusive(positions)
        )
        return shared, exclusive

    def forward(
        self,
        skills: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
        negative_flip_probability: float | None = None,
    ):
        questions_shared, questions_exclusive = self._question_embeddings(skills)
        interaction_ids = torch.where(
            mask, skills + responses * self.num_skills, torch.zeros_like(skills)
        )
        mean_s, var_s = self.encoder_shared(
            questions_shared, self.interaction_shared(interaction_ids), mask
        )
        mean_e, var_e = self.encoder_exclusive(
            questions_exclusive, self.interaction_exclusive(interaction_ids), mask
        )
        z_s = self._sample(mean_s, var_s)
        z_e = self._sample(mean_e, var_e)
        prediction = torch.sigmoid(
            self.predictor(torch.cat((z_s, z_e), dim=-1)).squeeze(-1)
        )
        result = {
            "prediction": prediction,
            "z_s": z_s,
            "mean_s": mean_s,
            "variance_s": var_s,
            "mean_e": mean_e,
            "variance_e": var_e,
        }

        if negative_flip_probability is not None:
            flipped = mask & (
                torch.rand(responses.shape, device=responses.device)
                < negative_flip_probability
            )
            negative_responses = torch.where(flipped, 1 - responses, responses)
            negative_ids = torch.where(
                mask,
                skills + negative_responses * self.num_skills,
                torch.zeros_like(skills),
            )
            negative_mean, negative_variance = self.encoder_shared(
                questions_shared, self.interaction_shared(negative_ids), mask
            )
            result["negative_z_s"] = self._sample(negative_mean, negative_variance)
        return result


class SimilarityDiscriminator(nn.Module):
    """Per-time MLP similarity D from Eq. (17)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.network(torch.cat((target, source), dim=-1)).squeeze(-1)
        )


class DisenKT(nn.Module):
    def __init__(
        self,
        num_skills: int,
        max_seq_len: int,
        batch_size: int,
        hidden_size: int,
        num_heads: int,
        gcn_layers: int,
        dropout: float,
        graphs: dict[str, torch.Tensor],
    ):
        super().__init__()
        kwargs = {
            "num_skills": num_skills,
            "max_seq_len": max_seq_len,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "gcn_layers": gcn_layers,
            "dropout": dropout,
        }
        self.domains = nn.ModuleDict(
            {
                name: DomainBranch(**kwargs, adjacency=adjacency)
                for name, adjacency in graphs.items()
            }
        )
        self.global_encoder = copy.deepcopy(
            next(iter(self.domains.values())).encoder_shared
        ).requires_grad_(False)
        for branch in self.domains.values():
            branch.encoder_shared.load_state_dict(self.global_encoder.state_dict())
        self.discriminators = nn.ModuleDict(
            {name: SimilarityDiscriminator(hidden_size) for name in graphs}
        )
        self.register_buffer(
            "global_shared_rep",
            torch.zeros(batch_size, max_seq_len, hidden_size),
        )
        self.register_buffer("has_global_shared_rep", torch.tensor(False))
        for name in graphs:
            self.register_buffer(f"{name}_step", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def load_global_shared_encoder(self) -> None:
        state = self.global_encoder.state_dict()
        for branch in self.domains.values():
            branch.encoder_shared.load_state_dict(state)

    @torch.no_grad()
    def aggregate_shared_state(
        self,
        representations: dict[str, torch.Tensor],
        sample_counts: dict[str, int],
    ) -> None:
        """Aggregate domain-local shared parameters and last-batch states."""
        total = sum(sample_counts.values())
        weights = {name: count / total for name, count in sample_counts.items()}
        for global_parameter, *local_parameters in zip(
            self.global_encoder.parameters(),
            *(branch.encoder_shared.parameters() for branch in self.domains.values()),
            strict=True,
        ):
            global_parameter.zero_()
            for name, parameter in zip(self.domains, local_parameters, strict=True):
                global_parameter.add_(parameter, alpha=weights[name])

        # The author's global state uses each domain's final local batch.
        # Zero-fill unused batch slots when those final batches differ in size.
        self.global_shared_rep.zero_()
        for name, representation in representations.items():
            self.global_shared_rep[: representation.size(0)].add_(
                representation, alpha=weights[name]
            )
        self.has_global_shared_rep.fill_(True)

    def forward_domain(
        self,
        domain: str,
        skills: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
        flip_probability: float,
    ):
        return self.domains[domain](
            skills, responses, mask, negative_flip_probability=flip_probability
        )

    def forward_target(
        self, skills: torch.Tensor, responses: torch.Tensor, mask: torch.Tensor
    ):
        return self.domains["target"](skills, responses, mask)
