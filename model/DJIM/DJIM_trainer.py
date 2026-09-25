"""Registered DJIM compositions with the framework's DKT and SAINT."""

import torch

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("DJIM-DKT")
class DJIMDKTConfig(ModelConfig):
    embedding_dim: int = 128
    psychology_heads: int = 3
    psychology_layers: int = 2
    dropout: float = 0.2
    q_learning_rate: float = 0.1
    q_discount: float = 0.9
    q_epsilon: float = 0.1
    epochs: int = 150
    learning_rate: float = 1e-3
    lr_decay: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 128


@register_model_config("DJIM-SAINT")
class DJIMSAINTConfig(DJIMDKTConfig):
    saint_heads: int = 8
    saint_blocks: int = 2
    epochs: int = 200
    batch_size: int = 64


class _DJIMTrainer(BaseTrainer):
    base_model_name: str

    def build_components(self, rc, data_src):
        from model.DJIM.DJIM_data import BEHAVIOR_COLUMNS, DJIMModelData
        from model.DJIM.DJIM_model import DJIM

        train_data, val_data, test_data = DJIMModelData(data_src).prepare_data(rc)
        metadata = data_src.get_metadata()
        m = rc.model
        model = DJIM(
            base_model=self.base_model_name,
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            seq_len=rc.data.max_seq_len,
            num_features=len(BEHAVIOR_COLUMNS),
            dim=m.embedding_dim,
            psychology_heads=m.psychology_heads,
            psychology_layers=m.psychology_layers,
            dropout=m.dropout,
            saint_heads=getattr(m, "saint_heads", 8),
            saint_blocks=getattr(m, "saint_blocks", 2),
            q_learning_rate=m.q_learning_rate,
            q_discount=m.q_discount,
            q_epsilon=m.q_epsilon,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=m.lr_decay)
            if m.lr_decay
            else None
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            lr_scheduler=scheduler,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def forward_pass(self, batch_data):
        skill, response, mask, question, behavior = (
            self._move_tensor_to_device(value) for value in batch_data
        )
        prediction = self.model(question, skill, response, behavior, mask)
        y_hat, y_label, _ = self._extract_valid_predictions(
            prediction, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def test_forward_pass(self, batch_data):
        skill, response, mask, group_id, true_label, question, _, behavior = (
            self._move_tensor_to_device(value) for value in batch_data
        )
        prediction = self.model(question, skill, response, behavior, mask)
        y_hat = prediction[mask]
        y_label = true_label[mask].float()
        groups = group_id[mask]
        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": groups,
        }


@register_trainer("DJIM-DKT")
class DJIMDKTTrainer(_DJIMTrainer):
    base_model_name = "DKT"


@register_trainer("DJIM-SAINT")
class DJIMSAINTTrainer(_DJIMTrainer):
    base_model_name = "SAINT"
