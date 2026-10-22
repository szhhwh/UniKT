"""DyGKT model implementation."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Cosine time encoder."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
            ).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = timestamps.unsqueeze(dim=2)
        return torch.cos(self.w(timestamps))


class TimeDualDecayEncoder(nn.Module):
    """Dual-decay time encoder."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True) -> None:
        super().__init__()
        self.time_dim = time_dim

        self.w_short = nn.Linear(1, time_dim)
        self.w_long = nn.Linear(1, time_dim)
        self.w_short.weight = nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
            ).reshape(time_dim, -1)
        )
        self.w_short.bias = nn.Parameter(torch.zeros(time_dim))
        self.w_long.weight = nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
            ).reshape(time_dim, -1)
        )
        self.w_long.bias = nn.Parameter(torch.zeros(time_dim))

        self.w_o = nn.Linear(time_dim, time_dim)
        self.w_o.weight = nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, 9, time_dim * time_dim, dtype=np.float32)
            ).reshape(time_dim, -1)
        )
        self.w_o.bias = nn.Parameter(torch.zeros(time_dim))

        self.f = nn.ReLU()

        if not parameter_requires_grad:
            self.w_short.weight.requires_grad = False
            self.w_short.bias.requires_grad = False
            self.w_long.weight.requires_grad = False
            self.w_long.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = timestamps.unsqueeze(dim=2)
        timestamps_right = torch.cat(
            [timestamps[:, 1:, :], timestamps[:, -1, :].unsqueeze(1)], dim=1
        )
        timestamps_diff = timestamps_right - timestamps

        mask = (timestamps_diff > 3600 * 24).float()

        short_features = self.f(self.w_short(timestamps_diff * mask))
        long_features = self.f(self.w_long(timestamps_diff * (1 - mask)))

        return self.w_o(short_features + long_features)


class MergeLayer(nn.Module):
    """Link predictor for DyGKT."""

    def __init__(
        self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_1, input_2], dim=1)
        return self.fc2(self.act(self.fc1(x)))


class DyKTSeq(nn.Module):
    """GRU updater block for DyGKT."""

    def __init__(self, edge_dim: int, node_dim: int) -> None:
        super().__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)
        self.hid_node_updater = nn.GRU(
            input_size=edge_dim, hidden_size=node_dim, batch_first=True
        )

    def update(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.hid_node_updater(x)
        return torch.squeeze(hidden, dim=0)


class DyGKT(nn.Module):
    """Dynamic Graph-based Knowledge Tracing model."""

    def __init__(
        self,
        data_metadata: dict[str, Any],
        *,
        num_neighbor: int = 50,
        ablation: str = "-1",
        dim_time: int = 16,
        edge_dim: int = 64,
        node_dim: int = 64,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.data_metadata = data_metadata

        self.num_neighbors = int(num_neighbor)
        self.ablation = str(ablation)
        self.time_dim = int(dim_time)
        self.edge_dim = int(edge_dim)
        self.node_dim = int(node_dim)

        num_questions = int(data_metadata["num_questions"])
        num_users = int(data_metadata["num_users"])
        self.question_id_offset = int(data_metadata.get("question_id_offset", 1))
        self.user_id_offset = int(
            data_metadata.get("user_id_offset", num_questions + self.question_id_offset)
        )
        self.num_nodes = self.user_id_offset + num_users

        question_features = np.asarray(
            data_metadata["question_features"], dtype=np.float32
        )

        node_feature_dim = int(question_features.shape[1])
        node_raw_features = np.zeros(
            (self.num_nodes, node_feature_dim), dtype=np.float32
        )
        node_raw_features[
            self.question_id_offset : self.question_id_offset + num_questions, :
        ] = question_features

        self.num_skills = int(data_metadata["num_skills"])

        self.register_buffer(
            "node_raw_features", torch.from_numpy(node_raw_features), persistent=False
        )
        self.projection_layer = nn.ModuleDict(
            {
                "feature_Linear": nn.Linear(
                    in_features=node_feature_dim, out_features=self.node_dim, bias=True
                ),
                "feature_Embed": nn.Embedding(
                    self.num_skills + 1, self.node_dim, padding_idx=0
                ),
                "node": nn.Embedding(self.num_nodes, self.node_dim),
                "edge": nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
                "time": nn.Linear(
                    in_features=self.time_dim, out_features=self.node_dim, bias=True
                ),
                "struct": nn.Linear(
                    in_features=1, out_features=self.node_dim, bias=True
                ),
            }
        )

        self.output_layer = nn.Linear(
            in_features=self.node_dim, out_features=self.node_dim, bias=True
        )
        self.dropout_layer = nn.Dropout(float(dropout))

        self.src_node_updater = DyKTSeq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        self.dst_node_updater = DyKTSeq(edge_dim=self.edge_dim, node_dim=self.node_dim)

        if self.ablation == "dual":
            self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

        self.link_predictor = MergeLayer(
            input_dim1=self.node_dim,
            input_dim2=self.node_dim,
            hidden_dim=self.node_dim,
            output_dim=1,
        )

    def set_neighbor_sampler(self, neighbor_sampler: Any) -> None:
        self.neighbor_sampler = neighbor_sampler

    def get_node_features(
        self,
        node_ids: torch.Tensor,
        skill_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.ablation in ["embed", "q_kid"]:
            features = self.projection_layer["feature_Embed"](skill_ids)
        elif self.ablation == "q_qid":
            features = self.projection_layer["node"](node_ids)
        else:
            raw = self.node_raw_features[node_ids].float()
            features = self.projection_layer["feature_Linear"](raw)

        if self.ablation == "skill":
            features = features * 0
        return features

    def get_edge_time_features(
        self,
        nodes_edge_features: torch.Tensor,
        nodes_neighbor_times: torch.Tensor,
        node_interact_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ablation == "dual":
            rel_time = node_interact_times.unsqueeze(1) - nodes_neighbor_times
            nodes_neighbor_time_features = self.time_encoder(rel_time)
        else:
            nodes_neighbor_time_features = self.time_encoder(nodes_neighbor_times)

        nodes_neighbor_time_features = self.projection_layer["time"](
            nodes_neighbor_time_features
        )
        nodes_edge_raw_features = self.projection_layer["edge"](nodes_edge_features)

        if self.ablation == "time":
            nodes_neighbor_time_features = nodes_neighbor_time_features * 0

        return (
            nodes_edge_raw_features,
            nodes_neighbor_time_features,
        )

    def compute_src_dst_node_temporal_embeddings(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_node_ids = batch["user"].long()
        dst_node_ids = batch["question"].long()
        node_interact_times = batch["time"].float()

        src_neighbor_node_ids = batch["src_neighbor_node_ids"].long()
        dst_neighbor_node_ids = batch["dst_neighbor_node_ids"].long()
        target_skill_ids = batch["skill"].long()
        src_neighbor_skill_ids = batch["src_neighbor_skill_ids"].long()

        src_neighbor_times = batch["src_neighbor_times"].float()
        dst_neighbor_times = batch["dst_neighbor_times"].float()

        src_neighbor_edge_feats = batch["src_neighbor_edge_feats"].float().unsqueeze(-1)
        dst_neighbor_edge_feats = batch["dst_neighbor_edge_feats"].float().unsqueeze(-1)

        batch_size = src_node_ids.shape[0]
        device = src_node_ids.device

        src_neighbor_times = torch.cat(
            [src_neighbor_times, node_interact_times.unsqueeze(1)], dim=1
        )
        dst_neighbor_times = torch.cat(
            [dst_neighbor_times, node_interact_times.unsqueeze(1)], dim=1
        )

        zero_edge = torch.zeros(batch_size, 1, 1, device=device)
        src_neighbor_edge_feats = torch.cat([src_neighbor_edge_feats, zero_edge], dim=1)
        dst_neighbor_edge_feats = torch.cat([dst_neighbor_edge_feats, zero_edge], dim=1)

        src_nodes_neighbor_co_occurrence_features = (
            (src_neighbor_node_ids == dst_node_ids.unsqueeze(1)).unsqueeze(-1).float()
        )
        dst_nodes_neighbor_co_occurrence_features = (
            (dst_neighbor_node_ids == src_node_ids.unsqueeze(1)).unsqueeze(-1).float()
        )

        src_nodes_neighbor_skill_features = (
            (src_neighbor_skill_ids == target_skill_ids.unsqueeze(1))
            .unsqueeze(-1)
            .float()
        )

        a = 0.0 if self.ablation == "counter" else 1.0

        struct_features = (
            torch.stack(
                [
                    src_nodes_neighbor_co_occurrence_features,
                    dst_nodes_neighbor_co_occurrence_features,
                    src_nodes_neighbor_skill_features,
                ],
                dim=0,
            )
            * a
        )
        struct_projections = self.projection_layer["struct"](struct_features)
        src_nodes_neighbor_struct_features = struct_projections[0]
        dst_nodes_neighbor_struct_features = struct_projections[1]
        src_nodes_neighbor_skill_struct_features = struct_projections[2]

        src_nodes_neighbor_node_raw_features = self.get_node_features(
            src_neighbor_node_ids, src_neighbor_skill_ids
        )
        dst_node_raw_features = self.get_node_features(dst_node_ids, target_skill_ids)
        src_nodes_edge_raw_features, src_nodes_neighbor_time_features = (
            self.get_edge_time_features(
                nodes_edge_features=src_neighbor_edge_feats,
                nodes_neighbor_times=src_neighbor_times,
                node_interact_times=node_interact_times,
            )
        )
        dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = (
            self.get_edge_time_features(
                nodes_edge_features=dst_neighbor_edge_feats,
                nodes_neighbor_times=dst_neighbor_times,
                node_interact_times=node_interact_times,
            )
        )

        src_edge_time_features = (
            src_nodes_edge_raw_features + src_nodes_neighbor_time_features
        )
        dst_edge_time_features = (
            dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features
        )
        src_nodes_features = (
            src_nodes_neighbor_node_raw_features
            + src_nodes_edge_raw_features[:, :-1, :]
            + src_nodes_neighbor_time_features[:, :-1, :]
        )

        src_node_embeddings = (
            self.src_node_updater.update(
                src_nodes_features
                + src_nodes_neighbor_skill_struct_features
                + src_nodes_neighbor_struct_features
            )
            + src_edge_time_features[:, -1, :]
        )

        if self.ablation in ["q_qid", "q_kid"]:
            dst_node_embeddings = dst_node_raw_features
        else:
            dst_target_features = (
                dst_node_raw_features
                + dst_nodes_edge_raw_features[:, -1, :]
                + dst_nodes_neighbor_time_features[:, -1, :]
            )
            dst_node_embeddings = (
                self.dst_node_updater.update(
                    dst_edge_time_features[:, :-1, :]
                    + dst_nodes_neighbor_struct_features
                )
                + dst_target_features
            )

        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)

        return src_node_embeddings, dst_node_embeddings

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        src_node_embeddings, dst_node_embeddings = (
            self.compute_src_dst_node_temporal_embeddings(batch)
        )

        src_node_embeddings = self.dropout_layer(src_node_embeddings)
        dst_node_embeddings = self.dropout_layer(dst_node_embeddings)

        logits = self.link_predictor(src_node_embeddings, dst_node_embeddings).squeeze(
            dim=-1
        )
        return logits
