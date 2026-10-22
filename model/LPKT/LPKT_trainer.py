"""LPKT / LPKT-S 模型训练器。

两模型共享 :class:`LPKTTrainerBase`（数据准备、优化器、前向与损失），
子类仅提供 ``_build_model``。数据侧共享
:class:`~model.LPKT.LPKT_data.LPKTModelData`（batch 含 uid，LPKT 忽略）。
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@dataclass
class LPKTConfigBase(ModelConfig):
    """Shared LPKT / LPKT-S model configuration.

    Args:
        hidden_size: Hidden dimension h.
        dropout: Dropout rate, applied to learn_gains only.
        q_gamma: Smoothing added to each event's skill indicator.
        max_it_minutes: Interval-time cap in minutes.
        learning_rate: Learning rate.
        weight_decay: Adam weight decay.
        max_grad_norm: Max gradient norm for clipping, 0 disables.
        lr_decay_step: StepLR step size.
        lr_decay_rate: StepLR decay factor.
        epochs: Max training epochs (early stopping applies).
        batch_size: Batch size.
    """

    hidden_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    q_gamma: float = 0.03
    max_it_minutes: int = 14400
    learning_rate: float = field(
        default=0.002,
        metadata={
            "optuna": {"type": "float", "low": 0.0005, "high": 0.005, "log": True}
        },
    )
    weight_decay: float = field(
        default=1e-6,
        metadata={"optuna": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True}},
    )
    max_grad_norm: float = 1.0
    lr_decay_step: int = 10
    lr_decay_rate: float = 0.5
    epochs: int = 100
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_model_config("LPKT")
class LPKTConfig(LPKTConfigBase):
    """LPKT model configuration."""


@register_model_config("LPKTS")
class LPKTSConfig(LPKTConfigBase):
    """LPKT-S model configuration."""


class LPKTTrainerBase(BaseTrainer):
    """LPKT / LPKT-S 公共训练器。"""

    def _build_model(self, info: dict, m) -> nn.Module:
        """子类构造各自的网络（LPKTNet / LPKTSNet）。"""
        raise NotImplementedError

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.LPKT.LPKT_data import LPKTModelData

        train_dataset, val_dataset, test_dataset, info = LPKTModelData(
            data_src
        ).prepare_data(rc)
        m = rc.model
        model = self._build_model(info, m)

        loss_fn = nn.BCELoss(reduction="none")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=m.lr_decay_step, gamma=m.lr_decay_rate
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            max_clip_grad_norm=m.max_grad_norm or None,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """模型输出 ``pred[:, t]`` 预测 ``a[:, t]``（same-position 约定）。

        uid 统一传入模型：LPKTS 用作 student embedding 索引，LPKT 忽略。
        """
        e, at, a, it, mask, uid, skills = batch_data
        e = self._move_tensor_to_device(e)
        at = self._move_tensor_to_device(at)
        a = self._move_tensor_to_device(a)
        it = self._move_tensor_to_device(it)
        mask = self._move_tensor_to_device(mask)
        uid = self._move_tensor_to_device(uid)
        skills = self._move_tensor_to_device(skills)

        pred = self.model(e, at, a, it, skills, uid)  # [B, S]

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


@register_trainer("LPKT")
class LPKTTrainer(LPKTTrainerBase):
    """LPKT 训练器。"""

    def _build_model(self, info: dict, m) -> nn.Module:
        from model.LPKT.LPKT_model import LPKTNet

        return LPKTNet(
            num_questions=info["num_questions"],
            num_skills=info["num_skills"],
            n_at=info["n_at"],
            n_it=info["n_it"],
            num_users=info["num_users"],
            hidden_size=m.hidden_size,
            dropout=m.dropout,
            q_gamma=m.q_gamma,
        )


@register_trainer("LPKTS")
class LPKTSTrainer(LPKTTrainerBase):
    """LPKT-S 训练器。"""

    def _build_model(self, info: dict, m) -> nn.Module:
        from model.LPKT.LPKT_model import LPKTSNet

        return LPKTSNet(
            num_questions=info["num_questions"],
            num_skills=info["num_skills"],
            n_at=info["n_at"],
            n_it=info["n_it"],
            num_users=info["num_users"],
            hidden_size=m.hidden_size,
            dropout=m.dropout,
            q_gamma=m.q_gamma,
        )
