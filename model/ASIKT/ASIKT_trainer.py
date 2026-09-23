"""ASIKT trainer for the UniKT next-item and windowlate protocols."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("ASIKT")
class ASIKTConfig(ModelConfig):
    d_model: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    n_blocks: int = field(
        default=2, metadata={"optuna": {"type": "int", "low": 1, "high": 4}}
    )
    n_heads: int = 8
    d_ff: int = 1024
    final_fc_dim: int = 512
    dropout: float = field(
        default=0.2, metadata={"optuna": {"type": "float", "low": 0.1, "high": 0.5}}
    )
    rasch: bool = True
    theta: float = 0.5
    d_state: int = 128
    chunk_size: int = 256
    epochs: int = 300
    learning_rate: float = field(
        default=5e-4,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True}},
    )
    weight_decay: float = 0.0
    batch_size: int = field(
        default=48,
        metadata={"optuna": {"type": "categorical", "choices": [32, 48, 64]}},
    )


@register_trainer("ASIKT")
class ASIKTTrainer(BaseTrainer):
    def build_components(self, rc, data_src):
        from model.ASIKT.ASIKT_data import ASIKTModelData
        from model.ASIKT.ASIKT_model import ASIKT

        model_data = ASIKTModelData(data_src)
        train, val, test = model_data.prepare_data(rc)
        metadata = data_src.get_metadata()
        m = rc.model
        model = ASIKT(
            n_skill=metadata["num_skills"],
            n_question=metadata["num_questions"],
            skill_prior=model_data.skill_prior,
            skill_frequency=model_data.skill_frequency,
            question_prior=model_data.question_prior,
            question_frequency=model_data.question_frequency,
            d_model=m.d_model,
            dropout=m.dropout,
            n_blocks=m.n_blocks,
            rasch=m.rasch,
            final_fc_dim=m.final_fc_dim,
            n_heads=m.n_heads,
            d_ff=m.d_ff,
            theta=m.theta,
            d_state=m.d_state,
            chunk_size=m.chunk_size,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            train_data=train,
            val_data=val,
            test_data=test,
        )

    def forward_pass(self, batch_data):
        skill, response, mask, question = (
            self._move_tensor_to_device(tensor) for tensor in batch_data
        )
        full = self.model(question, skill, response, mask)
        prediction, label, _ = self._extract_valid_predictions(full, response, mask)
        prediction, label = self._handle_empty_batch(prediction, label)
        return {
            "y_hat": prediction,
            "y_label": label,
            "y_predict": self._generate_binary_predictions(prediction, threshold=0.5),
            "y_score": prediction,
            "y_prob": prediction,
        }

    def test_forward_pass(self, batch_data):
        skill, response, mask, group, true_label, question, _ = (
            self._move_tensor_to_device(tensor) for tensor in batch_data
        )
        # Windowlate's mask selects the held-out target; group marks the
        # preceding visible interactions needed for attention and difficulty.
        valid = group >= 0
        full = self.model(question, skill, response, valid)
        target = mask[:, 1:].bool()
        prediction = torch.masked_select(full[:, :-1], target)
        label = torch.masked_select(true_label[:, 1:].float(), target)
        group_ids = torch.masked_select(group[:, 1:], target)
        return {
            "y_hat": prediction,
            "y_label": label,
            "y_predict": self._generate_binary_predictions(prediction, threshold=0.5),
            "y_score": prediction,
            "y_prob": prediction,
            "group_id": group_ids,
        }
