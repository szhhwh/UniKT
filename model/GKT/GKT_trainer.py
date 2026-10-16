"""GKT (Graph-based Knowledge Tracing) 训练器模块"""

from dataclasses import field
from typing import Literal

import numpy as np
import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


def build_dense_graph(num_c: int) -> torch.Tensor:
    """构建全连接图

    Args:
        num_c: 概念数量

    Returns:
        图邻接矩阵，形状为 [num_c, num_c]
    """
    graph = 1.0 / (num_c - 1) * np.ones((num_c, num_c))
    np.fill_diagonal(graph, 0)
    return torch.from_numpy(graph).float()


def build_transition_graph(sequences: list, num_c: int) -> torch.Tensor:
    """构建转移图

    基于概念序列的转移模式构建邻接矩阵

    Args:
        sequences: 概念序列列表
        num_c: 概念数量

    Returns:
        图邻接矩阵，形状为 [num_c, num_c]
    """
    graph = np.zeros((num_c, num_c))

    for seq in sequences:
        # Drop padding entries (negative ids)
        valid_seq = [s for s in seq if s >= 0 and s < num_c]
        for i in range(len(valid_seq) - 1):
            pre = valid_seq[i]
            next_c = valid_seq[i + 1]
            graph[pre, next_c] += 1

    np.fill_diagonal(graph, 0)

    rowsum = np.array(graph.sum(1))

    def inv(x):
        return 1.0 / x if x != 0 else 0.0

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)

    return torch.from_numpy(graph).float()


@register_model_config("GKT")
class GKTConfig(ModelConfig):
    """GKT model configuration.

    Args:
        hidden_dim: Hidden dimension of the model.
        embedding_dim: Embedding dimension of the model.
        dropout: Dropout probability.
        graph_type: Graph type for GKT model.
        use_gradient_checkpointing: Use per-timestep gradient checkpointing during
            training (extra recompute in backward for large activation-memory savings).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    hidden_dim: int = field(
        default=64,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    embedding_dim: int = field(
        default=64,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    dropout: float = field(
        default=0.5,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    graph_type: Literal["dense", "transition"] = "dense"
    use_gradient_checkpointing: bool = False
    epochs: int = 150
    learning_rate: float = field(
        default=0.001,
        metadata={
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=0.0001,
        metadata={
            "optuna": {"type": "float", "low": 1e-06, "high": 0.001, "log": True}
        },
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("GKT")
class GKTTrainer(BaseTrainer):
    """GKT 模型训练器

    负责初始化GKT模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.GKT.GKT_data import GKTModelData

        model_data = GKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        metadata = data_src.get_metadata()
        num_skills = metadata["num_skills"]

        m = rc.model
        graph = self._build_graph(m.graph_type, train_dataset, num_skills)

        from model.GKT.GKT_model import GKT

        logger.info("Initializing GKT model...")
        model = GKT(
            num_c=num_skills,
            hidden_dim=m.hidden_dim,
            emb_size=m.embedding_dim,
            graph_type=m.graph_type,
            graph=graph,
            dropout=m.dropout,
            use_gradient_checkpointing=m.use_gradient_checkpointing,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def _build_graph(
        self, graph_type: str, train_dataset, num_skills: int
    ) -> torch.Tensor:
        """构建图

        Args:
            graph_type: 图类型 ("dense" 或 "transition")
            train_dataset: 训练数据集
            num_skills: 技能数量

        Returns:
            图邻接矩阵
        """
        if graph_type == "dense":
            logger.info("Building dense graph...")
            return build_dense_graph(num_skills)
        elif graph_type == "transition":
            logger.info("Building transition graph from training data...")
            sequences = [
                train_dataset[i][0].tolist() for i in range(len(train_dataset))
            ]
            return build_transition_graph(sequences, num_skills)
        else:
            logger.warning(f"Unknown graph type: {graph_type}, using dense graph")
            return build_dense_graph(num_skills)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """GKT 前向传播

        Args:
            batch_data: 包含 (sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # Model outputs [B, S-1]; y[:, t] predicts response[:, t+1] (next-item alignment)
        y_hat_full = self._pad_to_full_sequence(self.model(sequence, response))

        # Pad to [B, S] then extract valid positions via built-in next-item alignment
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response,
            mask,
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估

        数据格式说明：
        - sequence: [技能历史, 目标技能]，形状 [B, S]
        - response: [历史标签, 0]  # 目标位置 response=0 避免数据泄露，形状 [B, S]
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测，形状 [B, S]
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id，形状 [B, S]
        - true_labels: [历史标签, 真实标签]  # 用于评估，形状 [B, S]

        GKT 预测语义：
        - 模型输出 [B, S-1]，其中 y[:, t] 预测 response[:, t+1]
        - 即 y[:, 0] 预测 response[:, 1], ..., y[:, S-2] 预测 response[:, S-1]
        """
        sequence, response, mask, late_group_id, true_labels, _, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        # Output [B, S-1]; y[:, t] predicts response[:, t+1]
        y_hat_full = self.model(sequence, response)

        # GKT prediction alignment: model outputs [B, S-1] where y[:, t]
        # predicts response[:, t+1], so y_hat_full aligns with [:, 1:].
        true_labels_aligned = true_labels[:, 1:]  # [B, S-1]
        mask_aligned = mask[:, 1:]  # [B, S-1]
        group_id_aligned = late_group_id[:, 1:]  # [B, S-1]

        y_hat = torch.masked_select(y_hat_full, mask_aligned)
        y_label = torch.masked_select(true_labels_aligned, mask_aligned).float()
        group_ids = torch.masked_select(group_id_aligned, mask_aligned)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
