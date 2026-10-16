from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv, Linear

from ..layers import GeneralInteraction, HGNNConv, HistoryRecap, Hypergraph


class HeteroGNN(nn.Module):
    """基于 HGT 的异质图神经网络模块。

    使用 Heterogeneous Graph Transformer 进行多层异构图聚合。

    Args:
        embedding_dim: 节点嵌入维度
        n_hop: GNN 层数
        heads: 注意力头数
        dropout: Dropout 概率
        metadata: 异构图元数据

    Example:
        >>> gnn = HeteroGNN(embedding_dim=128, n_hop=2, heads=4, dropout=0.2, metadata=metadata)
        >>> output = gnn(x_dict, edge_index_dict)
    """

    def __init__(
        self,
        embedding_dim: int,
        n_hop: int,
        heads: int,
        dropout: float,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        output_node_types: tuple[str, ...] | None = ("question", "skill"),
    ) -> None:
        super().__init__()
        self.n_hop = n_hop
        self.heads = heads
        self.dropout = dropout
        self.output_node_types = output_node_types
        self.convs = torch.nn.ModuleList()

        for _ in range(n_hop):
            conv = HGTConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                metadata=metadata,
                heads=heads,
            )
            self.convs.append(conv)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典

        Returns:
            聚合后的节点表示字典
        """
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            is_last = i == self.n_hop - 1
            new_x_dict = {}
            for node_type, x in x_dict.items():
                if x is not None:
                    needs_post = (
                        self.output_node_types is None
                        or node_type in self.output_node_types
                        or not is_last
                    )
                    if needs_post:
                        x = F.gelu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[node_type] = x
            x_dict = new_x_dict
        return x_dict


class HyperGNN(nn.Module):
    """双层 HGNN 超图神经网络

    使用 ``model.layers.hypergraph.HGNNConv`` 实现双层超图卷积，支持加权超图。

    数学公式：
        X' = σ(D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ)

    其中：
        - X 是输入顶点特征矩阵
        - H 是超图关联矩阵
        - W_e 是超边权重对角矩阵（可自定义或默认为单位矩阵）
        - D_v 是顶点度数对角矩阵
        - D_e 是超边度数对角矩阵
        - Θ 是可学习参数
    """

    def __init__(
        self,
        in_ch: int,
        n_hid: int,
        n_class: int,
        dropout: float = 0.0,
        use_edge_weights: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge_weights = use_edge_weights

        # First conv layer: aggregates direct-neighbor question features
        # is_last=False enables the built-in activation and dropout
        self.hgc1 = HGNNConv(
            in_ch, n_hid, bias=True, use_bn=False, drop_rate=dropout, is_last=False
        )
        # Second conv layer: aggregates indirect-neighbor question features
        # is_last=True skips the built-in activation/dropout; ReLU is applied manually in forward
        self.hgc2 = HGNNConv(
            n_hid, n_class, bias=True, use_bn=False, drop_rate=dropout, is_last=True
        )

    def forward(self, x: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征矩阵 [num_vertices, in_ch]
            hg: 超图结构

        Returns:
            输出特征矩阵 [num_vertices, n_class]
        """
        x1 = self.hgc1(x, hg)
        x2 = F.relu(self.hgc2(x1, hg))

        return x2


class MoEFusion(nn.Module):
    """混合专家融合 (Mixture-of-Experts Fusion)。

    将两个视图的特征处理视为不同的"专家"。
    引入一个共享专家(Shared Expert)捕获共性。
    使用门控网络(Router)动态分配权重。
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        # Expert networks:
        # Expert 1: processes View 1
        # Expert 2: processes View 2
        # Expert 3: processes View 1 + View 2 (shared)
        self.expert1 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.expert2 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.expert_shared = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout)
        )

        # Router: takes both views and outputs per-expert weights
        self.router = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        # Flatten batch dimensions for Linear layers
        B_shape = view1.shape[:-1]
        v1_flat = view1.reshape(-1, self.dim)
        v2_flat = view2.reshape(-1, self.dim)

        e1 = self.expert1(v1_flat)
        e2 = self.expert2(v2_flat)
        combined = torch.cat([v1_flat, v2_flat], dim=-1)
        e_shared = self.expert_shared(combined)

        # Stack expert outputs: [N, 3, D]
        experts = torch.stack([e1, e2, e_shared], dim=1)

        weights = self.router(combined)  # [N, 3]

        # weights: [N, 3] -> [N, 3, 1]
        # experts: [N, 3, D]
        fused = torch.sum(experts * weights.unsqueeze(-1), dim=1)

        return self.norm(fused).reshape(*B_shape, self.dim)


class HDHKT(nn.Module):
    """HDHKT 主模型。

    层次化图知识追踪模型，融合异构图和超图进行预测。

    Args:
        data_metadata: 数据集元数据
        hetero_metadata: 异构图元数据
        hidden_dim: 隐藏层维度
        n_hop: GNN 层数
        heads: 注意力头数
        lstm_layers: LSTM 层数
        dropout: Dropout 概率（所有层共享）
        history_neighbour: 历史邻居数量
        att_bound: 注意力边界
        **kwargs: 额外的关键字参数

    Example:
        >>> model = HDHKT(data_metadata, hetero_metadata, hidden_dim=250, n_hop=4, heads=1,
        ...               lstm_layers=1, dropout=0.25, history_neighbour=5, att_bound=0.1)
        >>> logits = model(user_sequence, user_response, user_mask, hetero_graph, hypergraph, interaction_skills)
    """

    def __init__(
        self,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        *,
        hidden_dim: int,
        n_hop: int,
        heads: int,
        lstm_layers: int,
        dropout: float,
        history_neighbour: int,
        att_bound: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_metadata = data_metadata

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        self.question_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.question_embedding_hyper = torch.nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.skill_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_skills"],
            embedding_dim=self.hidden_dim,
        )
        self.assignment_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_assignments"],
            embedding_dim=self.hidden_dim,
        )
        self.template_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_templates"],
            embedding_dim=self.hidden_dim,
        )
        self.answer_embedding = torch.nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.hidden_dim,
        )
        self.embedding_dropout = torch.nn.Dropout(p=self.dropout)

        self.hetero_conv = HeteroGNN(
            embedding_dim=self.hidden_dim,
            n_hop=n_hop,
            heads=heads,
            dropout=self.dropout,
            metadata=hetero_metadata,
        )

        self.hgnn_conv = HyperGNN(
            in_ch=self.hidden_dim,
            n_hid=self.hidden_dim,
            n_class=self.hidden_dim,
            dropout=self.dropout,
        )

        self.fuse = MoEFusion(dim=self.hidden_dim, dropout=self.dropout)

        self.fc_exercise = Linear(
            self.hidden_dim * 2, self.hidden_dim, weight_initializer="uniform"
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

        # Eval-mode graph-backbone cache.
        self._graph_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def train(self, mode: bool = True) -> "HDHKT":
        """Enter train/eval mode, invalidating the eval GNN cache on switch."""
        prev = self.training
        out = super().train(mode)
        if mode != prev:
            self._graph_cache.clear()
        return out

    def _compute_graph_outputs(
        self,
        hetero_graph: Any,
        hypergraph: Hypergraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the graph backbone.

        Returns:
            (question_conv_fused [num_questions, H], skill_hetero_conv [num_skills, H]).
        """
        question_hyper_conv: torch.Tensor = self.hgnn_conv(
            self.question_embedding_hyper.weight, hypergraph
        )
        conv = self.hetero_conv(
            {
                "question": self.question_embedding.weight,
                "skill": self.skill_embedding.weight,
                "assignment": self.assignment_embedding.weight,
                "template": self.template_embedding.weight,
            },
            hetero_graph.edge_index_dict,
        )
        question_hetero_conv: torch.Tensor = conv["question"]
        skill_hetero_conv: torch.Tensor = conv["skill"]
        question_conv_fused = self.fuse(question_hetero_conv, question_hyper_conv)
        return question_conv_fused, skill_hetero_conv

    def _cached_graph_outputs(
        self,
        hetero_graph: Any,
        hypergraph: Hypergraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached eval-mode graph outputs, computing once per graph pair."""
        key = (id(hetero_graph), id(hypergraph))
        cached = self._graph_cache.get(key)
        if cached is None:
            with torch.no_grad():
                question_conv_fused, skill_hetero_conv = self._compute_graph_outputs(
                    hetero_graph, hypergraph
                )
            cached = {
                "question_conv_fused": question_conv_fused,
                "skill_hetero_conv": skill_hetero_conv,
            }
            self._graph_cache[key] = cached
        return cached["question_conv_fused"], cached["skill_hetero_conv"]

    def forward(
        self,
        user_sequence: torch.Tensor,  # [B, S]
        user_response: torch.Tensor,  # [B, S]
        user_mask: torch.Tensor,  # [B, S]
        hetero_graph: Any,  # HeteroData
        hypergraph: Hypergraph,
        interaction_skills: torch.Tensor,  # [B, S, K_max], padded with num_skills
        return_states: bool = False,
    ) -> torch.Tensor:  # [B, S]
        """前向传播。

        Args:
            user_sequence: 用户问题序列 [B, S]
            user_response: 用户回答序列 [B, S]
            user_mask: 有效位置掩码 [B, S]
            hetero_graph: 异构图数据
            hypergraph: 超图数据
            interaction_skills: 本次交互的技能 id 列表 [B, S, K_max]，
                未用槽位填 ``num_skills``（前向时追加零向量行）
            return_states: 是否返回内部状态（用于知识状态计算）

        Returns:
            预测 logits [B, S]，或 (logits, skill_hetero_conv, lstm_output) 当 return_states=True
        """
        B, _ = user_sequence.size()

        # [B, S, embedding_dim]
        answers_embedding: torch.Tensor = self.answer_embedding(user_response)

        if self.training:
            question_conv_fused, skill_hetero_conv = self._compute_graph_outputs(
                hetero_graph, hypergraph
            )
        else:
            question_conv_fused, skill_hetero_conv = self._cached_graph_outputs(
                hetero_graph, hypergraph
            )  # [num_questions, H] / [num_skills, H]

        question_embedding_sequence = question_conv_fused[user_sequence]

        exercise_emb = torch.cat(
            [question_embedding_sequence, answers_embedding], dim=-1
        )  # [B, S, 2*E]

        exercise_emb = F.relu(self.fc_exercise(exercise_emb))  # [B, S, H]
        exercise_emb = self.embedding_dropout(exercise_emb)

        # [B, S, H]
        lstm_output, _ = self.lstm(exercise_emb)

        # Shift to next-question sequence; last timestep is zero-padded
        next_user_sequence = torch.cat(
            [user_sequence[:, 1:], user_sequence.new_zeros(B, 1)], dim=1
        )  # [B, S]

        # [B, S, embedding_dim]
        next_question_embedding: torch.Tensor = question_conv_fused[next_user_sequence]

        history_question_neighbors = self.history_review(
            question_embedding_sequence,
            next_question_embedding,
            exercise_emb,
            user_mask,
        )  # [B, S, M, H]

        # Student status = LSTM output + history neighbors
        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )  # [B, S, M+1, H]

        # Knowledge status = next-question features + related skill features
        num_skills = skill_hetero_conv.size(0)
        skill_ids_full = torch.cat(
            [interaction_skills[:, 1:], interaction_skills[:, -1:]], dim=1
        )
        skill_slot_mask = skill_ids_full != num_skills

        # Append a zero-vector row so the padding index resolves to zeros
        skill_conv_padded = F.pad(skill_hetero_conv, (0, 0, 0, 1))  # [num_skills+1, H]

        # [B, S, K_max, embedding_dim]
        related_skill_embs = skill_conv_padded[skill_ids_full]

        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs],
            dim=2,
        )  # [B, S, K_max+1, embedding_dim]

        logits = self.general_interaction(
            student_status,
            knowledge_status,
            user_mask,
            skill_slot_mask=skill_slot_mask,
        )  # [B, S]

        if return_states:
            return logits, skill_hetero_conv, lstm_output
        return logits  # [B, S]
