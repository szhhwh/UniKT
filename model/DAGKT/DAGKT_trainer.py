"""DAGKT 模型训练器。

定义 DAGKT 模型特定的训练逻辑，包括组合损失函数。
"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["DAGKTTrainer", "DAGKTConfig"]


@register_model_config("DAGKT")
class DAGKTConfig(ModelConfig):
    """DAGKT model configuration.

    Args:
        hidden_dim: Hidden layer dimension.
        embedding_dim: Embedding dimension.
        lstm_layers: Number of LSTM layers.
        n_hop: Number of GNN hops.
        heads: Number of attention heads.
        history_neighbour: History neighbor count.
        att_bound: Attention bound.
        dropout: Dropout rate.
        ae_hidden_dim: Autoencoder hidden layer dimension.
        loss_diff_weight: Weight for difficulty autoencoder reconstruction loss.
        loss_attempt_weight: Weight for attempt autoencoder reconstruction loss.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    hidden_dim: int = 100
    embedding_dim: int = 100
    lstm_layers: int = 2
    n_hop: int = field(
        default=3,
        metadata={"optuna": {"type": "int", "low": 1, "high": 5}},
    )
    heads: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    history_neighbour: int = 5
    att_bound: float = 0.2
    dropout: float = field(
        default=0.4,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    ae_hidden_dim: int = 50
    loss_diff_weight: float = 1.0
    loss_attempt_weight: float = 1.0
    epochs: int = 150
    learning_rate: float = field(
        default=0.001,
        metadata={
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=1e-4,
        metadata={
            "optuna": {"type": "float", "low": 0.000001, "high": 0.001, "log": True}
        },
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("DAGKT")
class DAGKTTrainer(BaseTrainer):
    """DAGKT 模型训练器。

    在 GIKT 训练基础上增加辅助损失（难度和尝试次数的自编码器重建损失）。
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DAGKT.DAGKT_data import DAGKTModelData

        model_data = DAGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.graph,
            question_difficulty,
        ) = model_data.prepare_data(rc)

        from model.DAGKT.DAGKT_model import DAGKT

        logger.info("Initializing DAGKT model...")
        m = rc.model
        model = DAGKT(
            data_src.get_metadata(),
            question_difficulty,
            embedding_dim=m.embedding_dim,
            hidden_dim=m.hidden_dim,
            lstm_layers=m.lstm_layers,
            dropout=m.dropout,
            ae_hidden_dim=m.ae_hidden_dim,
            n_hop=m.n_hop,
            heads=m.heads,
            history_neighbour=m.history_neighbour,
            att_bound=m.att_bound,
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        self.loss_diff_weight = m.loss_diff_weight
        self.loss_attempt_weight = m.loss_attempt_weight

        dev = rc.general.device
        device = (
            torch.device(dev)
            if dev
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.graph = self.graph.to(device)

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self,
        batch_data: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> dict[str, torch.Tensor]:
        """DAGKT 前向传播。

        Args:
            batch_data: (sequence, response, mask, attempt_counts, skills)

        Returns:
            包含 y_hat, y_label, y_predict, _ae_loss_diff, _ae_loss_attempt 的字典
        """
        sequence, response, mask, attempt_counts, skills = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        skills = self._move_tensor_to_device(skills)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        attempt_counts = self._move_tensor_to_device(attempt_counts, dtype=torch.float)

        y_hat_full, loss_diff, loss_attempt = self.model(
            sequence,
            response,
            mask,
            self.graph,
            skills,
            attempt_counts,
        )  # y_hat_full: [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
            "_ae_loss_diff": loss_diff,
            "_ae_loss_attempt": loss_attempt,
        }

    def _compute_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """计算组合损失: BCE + difficulty_ae_loss + attempt_ae_loss。"""
        base_loss = super()._compute_loss(outputs)

        loss_diff = outputs.get("_ae_loss_diff", torch.tensor(0.0))
        loss_attempt = outputs.get("_ae_loss_attempt", torch.tensor(0.0))

        total_loss = (
            base_loss
            + self.loss_diff_weight * loss_diff
            + self.loss_attempt_weight * loss_attempt
        )

        return total_loss
