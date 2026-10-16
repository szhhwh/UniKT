"""HGKT 模型训练器。"""

from dataclasses import field

import torch
from torch import nn

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("HGKT")
class HGKTConfig(ModelConfig):
    """HGKT model configuration.

    Args:
        emb_dim: Raw exercise embedding dimension de (hypergraph encoder
            input, reused by the predict layer).
        hidden_size: Knowledge-state / gate dimension dk.
        time_dim: Answer- and interval-time embedding dimension da.
        dropout: Dropout rate, applied to learn_gains only.
        q_gamma: Smoothing added to each event's skill indicator.
        max_it_minutes: Interval-time cap in minutes.
        learning_rate: Learning rate.
        weight_decay: Adam weight decay.
        max_grad_norm: Max gradient norm for clipping, 0 disables.
        epochs: Max training epochs (early stopping applies).
        batch_size: Batch size.

    Note: no lr scheduler — the paper fixes lr at 3e-3 and mentions none,
    unlike LPKT's StepLR.
    """

    emb_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    hidden_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128]}},
    )
    time_dim: int = 50
    dropout: float = field(
        default=0.25,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    q_gamma: float = 0.03
    max_it_minutes: int = 14400
    learning_rate: float = field(
        default=0.0018,
        metadata={
            "optuna": {"type": "float", "low": 0.0005, "high": 0.005, "log": True}
        },
    )
    weight_decay: float = field(
        default=0.00046,
        metadata={"optuna": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True}},
    )
    max_grad_norm: float = 1.0
    epochs: int = 100
    batch_size: int = field(
        default=32,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("HGKT")
class HGKTTrainer(BaseTrainer):
    """HGKT 训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.HGKT.HGKT_data import HGKTModelData
        from model.HGKT.HGKT_model import HGKTNet

        train_dataset, val_dataset, test_dataset, info = HGKTModelData(
            data_src
        ).prepare_data(rc)
        m = rc.model
        model = HGKTNet(
            num_questions=info["num_questions"],
            num_skills=info["num_skills"],
            n_at=info["n_at"],
            n_it=info["n_it"],
            hyper_factors=info["hyper_factors"],
            emb_dim=m.emb_dim,
            hidden_size=m.hidden_size,
            time_dim=m.time_dim,
            dropout=m.dropout,
            q_gamma=m.q_gamma,
        )

        loss_fn = nn.BCELoss(reduction="none")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_clip_grad_norm=m.max_grad_norm or None,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """模型输出 ``pred[:, t]`` 预测 ``a[:, t]``"""
        e, at, a, it, mask, _, skills = batch_data
        e = self._move_tensor_to_device(e)
        at = self._move_tensor_to_device(at)
        a = self._move_tensor_to_device(a)
        it = self._move_tensor_to_device(it)
        mask = self._move_tensor_to_device(mask)
        skills = self._move_tensor_to_device(skills)

        pred = self.model(e, at, a, it, skills)  # [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(
            pred, a, mask, same_position=True
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

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """BCE 逐元素后按 batch 内所有有效位置求和。"""
        y_hat = outputs["y_hat"].clamp(1e-7, 1.0 - 1e-7)
        return self.loss(y_hat, outputs["y_label"]).sum()
