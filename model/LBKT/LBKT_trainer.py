from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["LBKTTrainer", "LBKTConfig"]


@register_model_config("LBKT")
class LBKTConfig(ModelConfig):
    """LBKT model configuration.

    Args:
        dim_tp: Topic (question) embedding dimension (default: 128).
        dim_hidden: Response embedding dimension (default: 50).
        num_units: Hidden dimension (default: 128).
        dropout: Dropout rate (default: 0.2).
        q_gamma: Smoothing added to each event's skill indicator (default: 0.1).
        epochs: Number of training epochs (default: 100).
        learning_rate: Learning rate for optimizer (default: 0.001).
        lr_decay_step: Learning rate decay step size (default: 1).
        lr_decay_rate: Learning rate decay factor (default: 0.5).
        weight_decay: Weight decay for optimizer (default: 1e-6).
        batch_size: Batch size for training (default: 16).
    """

    dim_tp: int = field(
        default=128,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    dim_hidden: int = 50
    num_units: int = field(
        default=128,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    q_gamma: float = 0.1
    epochs: int = 100
    learning_rate: float = field(
        default=0.001,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    lr_decay_step: int = 1
    lr_decay_rate: float = 0.5
    weight_decay: float = field(
        default=1e-6,
        metadata={"optuna": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=16,
        metadata={"optuna": {"type": "categorical", "choices": [16, 32, 64]}},
    )


@register_trainer("LBKT")
class LBKTTrainer(BaseTrainer):
    """LBKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.LBKT.LBKT_data import LBKTModelData
        from model.LBKT.LBKT_model import LBKT

        model_data = LBKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
        ) = model_data.prepare_data(rc)

        logger.info("Initializing LBKT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = LBKT(
            dim_tp=m.dim_tp,
            dim_hidden=m.dim_hidden,
            num_units=m.num_units,
            dropout=m.dropout,
            data_metadata=metadata,
            q_gamma=m.q_gamma,
        )

        # 优化器和损失函数
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
            eps=1e-8,
            betas=(0.1, 0.999),
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=m.lr_decay_step, gamma=m.lr_decay_rate
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

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """LBKT 前向传播。

        Args:
            batch_data: 包含 (sequence, response, mask, time_factor, attempt_factor, hint_factor) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, time_factor, attempt_factor, hint_factor, skills = (
            batch_data
        )

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        time_factor = self._move_tensor_to_device(time_factor)
        attempt_factor = self._move_tensor_to_device(attempt_factor)
        hint_factor = self._move_tensor_to_device(hint_factor)
        skills = self._move_tensor_to_device(skills)

        preds = self.model(
            sequence,
            response,
            time_factor,
            attempt_factor,
            hint_factor,
            skills,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            preds, response, mask, same_position=True
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
