"""MIKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("MIKT")
class MIKTConfig(ModelConfig):
    """MIKT 模型配置。

    Args:
        embed_dim: Embedding dimension (default: 64).
        state_dim: State representation dimension (default: 64).
        dropout: Dropout rate (default: 0.4).
        grad_clip: Gradient clipping norm (default: 15.0).
        epochs: Number of training epochs (default: 200).
        learning_rate: Learning rate for optimizer (default: 0.002).
        weight_decay: Weight decay for optimizer (default: 1e-5).
        batch_size: Batch size for training (default: 80).
        lr_decay: Learning rate decay factor per epoch.
    """

    embed_dim: int = field(
        default=64,
        metadata={"optuna": {"type": "int", "low": 32, "high": 128}},
    )
    # must equal embed_dim (assert in MIKT_model), so it follows embed_dim
    state_dim: int = field(default=64)
    dropout: float = field(
        default=0.4,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    grad_clip: float = 15.0
    epochs: int = 200
    learning_rate: float = field(
        default=0.002,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    weight_decay: float = field(
        default=1e-5,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=80,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 80, 128]}},
    )
    lr_decay: float | None = None


@register_trainer("MIKT")
class MIKTTrainer(BaseTrainer):
    """MIKT 模型训练器"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.MIKT.MIKT_data import MIKTModelData
        from model.MIKT.MIKT_model import MIKT

        logger.info("Initializing MIKT model...")
        m = rc.model
        metadata = data_src.get_metadata()
        model = MIKT(
            embed_dim=m.embed_dim,
            state_dim=m.embed_dim,
            dropout=m.dropout,
            data_metadata=metadata,
        )

        model_data = MIKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
        ) = model_data.prepare_data(rc)

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
            max_clip_grad_norm=m.grad_clip,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """MIKT 前向传播

        模型输出 P[:, t] 预测 response[:, t+1]，即 [B, S-1] 的预测。
        对应标签为 response[:, 1:]，对应掩码为 mask[:, 1:]。
        """
        sequence, response, mask, skills = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        skills = self._move_tensor_to_device(skills)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # Model outputs [B, S-1]; P[:, t] predicts response[:, t+1] (next-item). Pad to [B, S] for built-in alignment.
        y_hat_full = self._pad_to_full_sequence(
            self.model(sequence, response, mask, skills)
        )

        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

        return result
