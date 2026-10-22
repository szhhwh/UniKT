"""HDKT 模型训练器。"""

from dataclasses import field

import torch
from torch import nn

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("HDKT")
class HDKTConfig(ModelConfig):
    """HDKT model configuration.

    Args:
        hidden_size: Hidden dimension h.
        dropout: Dropout rate, shared by learn_gains and the discriminator.
        emb_dropout: Embedding dropout rate in the denoising components.
        tau: Hard gumbel-softmax temperature (not searched).
        vae_alpha: Weight of the VAE reconstruction loss.
        q_gamma: Smoothing added to each event's skill indicator.
        max_it_minutes: Interval-time cap in minutes.
        learning_rate: Learning rate.
        weight_decay: Adam weight decay.
        max_grad_norm: Max gradient norm for clipping, 0 disables.
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
    emb_dropout: float = field(
        default=0.3,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    tau: float = 100.0
    vae_alpha: float = field(
        default=1.0,
        metadata={"optuna": {"type": "float", "low": 0.001, "high": 1.0, "log": True}},
    )
    q_gamma: float = 0.0
    max_it_minutes: int = 14400
    learning_rate: float = field(
        default=0.001,
        metadata={
            "optuna": {"type": "float", "low": 0.0005, "high": 0.005, "log": True}
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True}},
    )
    max_grad_norm: float = 1.0
    epochs: int = 100
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("HDKT")
class HDKTTrainer(BaseTrainer):
    """HDKT 训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.HDKT.HDKT_data import HDKTModelData
        from model.HDKT.HDKT_model import HDKTNet

        train_dataset, val_dataset, test_dataset, info = HDKTModelData(
            data_src
        ).prepare_data(rc)
        m = rc.model
        model = HDKTNet(
            num_questions=info["num_questions"],
            num_skills=info["num_skills"],
            n_at=info["n_at"],
            n_it=info["n_it"],
            num_users=info["num_users"],
            max_seq_len=info["max_question_seq_len"],
            hidden_size=m.hidden_size,
            dropout=m.dropout,
            tau=m.tau,
            emb_dropout=m.emb_dropout,
            q_gamma=m.q_gamma,
        )

        # Mean reduction keeps the default _compute_eval_loss scalar-valued;
        # the "none" variant would return a vector there (item() crash).
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            max_clip_grad_norm=m.max_grad_norm or None,
        )

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """模型输出 ``pred[:, t]`` 预测 ``a[:, t]``（same-position 约定）。

        ``_vae_loss`` 为去噪前端的 VAE 重构损失，仅参与训练损失组合。
        """
        e, at, a, it, mask, uid, skill, skills = batch_data
        e = self._move_tensor_to_device(e)
        at = self._move_tensor_to_device(at)
        a = self._move_tensor_to_device(a)
        it = self._move_tensor_to_device(it)
        mask = self._move_tensor_to_device(mask)
        uid = self._move_tensor_to_device(uid)
        skill = self._move_tensor_to_device(skill)
        skills = self._move_tensor_to_device(skills)

        pred, vae_loss = self.model(e, at, a, it, mask, uid, skill, skills)  # [B, S]

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
            "_vae_loss": vae_loss,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """BCE 均值 + ``vae_alpha``·重构损失；val/test 走默认纯 BCE。"""
        y_hat = outputs["y_hat"].clamp(1e-7, 1.0 - 1e-7)
        return (
            self.loss(y_hat, outputs["y_label"])
            + self.run_config.model.vae_alpha * outputs["_vae_loss"]
        )
