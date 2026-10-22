from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["BDGKTTrainer", "BDGKTConfig"]


@register_model_config("BDGKT")
class BDGKTConfig(ModelConfig):
    """BDGKT model configuration.

    Args:
        hidden_dim: Hidden layer dimension.
        layer_num: Number of GNN layers.
        drop1: Feature dropout rate.
        drop2: Attention dropout rate.
        question_max_length: Student history length l_s.
        student_max_length: Top similar students per question l_q.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        batch_size: Batch size.
        epochs: Number of training epochs.
    """

    hidden_dim: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    layer_num: int = field(
        default=2,
        metadata={"optuna": {"type": "int", "low": 1, "high": 4}},
    )
    drop1: float = field(
        default=0.3,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    drop2: float = field(
        default=0.3,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    question_max_length: int = 20
    student_max_length: int = 5
    learning_rate: float = field(
        default=0.001,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=2000,
        metadata={"optuna": {"type": "categorical", "choices": [1000, 2000, 4000]}},
    )
    epochs: int = 100


@register_trainer("BDGKT")
class BDGKTTrainer(BaseTrainer):
    """BDGKT trainer using precomputed fixed-shape context tensors."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.BDGKT.BDGKT_data import BDGKTModelData, _collate_fn
        from model.BDGKT.BDGKT_model import BDGKT

        logger.info("Initializing BDGKT model...")

        # 1. prepare data
        model_data = BDGKTModelData(data_src, cache=rc.general.cache)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_students,
            num_questions,
        ) = model_data.prepare_data(rc)

        # 2. create model
        m = rc.model
        metadata = data_src.get_metadata()
        model = BDGKT(
            student_num=num_students,
            question_num=num_questions,
            skill_num=metadata["num_skills"],
            hidden_size=m.hidden_dim,
            student_max_length=m.student_max_length,
            question_max_length=m.question_max_length,
            drop1=m.drop1,
            drop2=m.drop2,
            layer_num=m.layer_num,
        )

        # 3. optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )
        loss_fn = torch.nn.BCELoss()

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=_collate_fn,
        )

    def forward_pass(self, batch_data):
        """BDGKT forward pass.

        Args:
            batch_data: (target_students, target_questions, labels,
                        target_skills, hist_skills,
                        hist_q, hist_r, hist_mask,
                        q_ans_s, q_ans_r, q_ans_mask)
        """
        (
            target_students,
            target_questions,
            labels,
            target_skills,
            hist_skills,
            hist_q,
            hist_r,
            hist_mask,
            q_ans_s,
            q_ans_r,
            q_ans_mask,
        ) = batch_data

        target_students = self._move_tensor_to_device(target_students)
        target_questions = self._move_tensor_to_device(target_questions)
        y_label = self._move_tensor_to_device(labels).float()
        target_skills = self._move_tensor_to_device(target_skills)
        hist_skills = self._move_tensor_to_device(hist_skills)

        hist_q = self._move_tensor_to_device(hist_q)
        hist_r = self._move_tensor_to_device(hist_r)
        hist_mask = self._move_tensor_to_device(hist_mask)
        q_ans_s = self._move_tensor_to_device(q_ans_s)
        q_ans_r = self._move_tensor_to_device(q_ans_r)
        q_ans_mask = self._move_tensor_to_device(q_ans_mask)

        pred = self.model(
            target_students,
            target_questions,
            target_skills,
            hist_skills,
            hist_q,
            hist_r,
            hist_mask,
            q_ans_s,
            q_ans_r,
            q_ans_mask,
        )

        if y_label.numel() == 0:
            raise ValueError("Empty valid targets in current batch.")

        y_predict = (pred >= 0.5).long()

        return {
            "y_hat": pred,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": pred,
            "y_prob": pred,
        }
