"""TFKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("TFKT")
class TFKTConfig(ModelConfig):
    """TFKT model configuration.

    Args:
        d_model: Hidden dimension of the model.
        n_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: Feed-forward network dimension.
        final_fc_dim: Final fully connected layer dimension.
        kq_same: Whether key and query share the linear transformation (1=yes, 0=no).
        separate_qa: Whether to use separate QA embeddings (1=yes, 0=no).
        use_rasch: Whether to enable the Rasch problem-id difficulty model (True=yes).
        l2: L2 regularization coefficient.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    n_blocks: int = field(
        default=1,
        metadata={"optuna": {"type": "int", "low": 1, "high": 4}},
    )
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    dropout: float = field(
        default=0.05,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    d_ff: int = 2048
    final_fc_dim: int = 512
    kq_same: int = 1
    separate_qa: int = 0
    use_rasch: bool = True
    l2: float = field(
        default=1e-5,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    epochs: int = 1000
    learning_rate: float = field(
        default=1e-5,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=0.0,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]}
        },
    )
    batch_size: int = field(
        default=24,
        metadata={"optuna": {"type": "categorical", "choices": [24, 32, 64]}},
    )


@register_trainer("TFKT")
class TFKTTrainer(BaseTrainer):
    """Build and train TFKT with the framework's prediction alignment."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.TFKT.TFKT_data import TFKTModelData

        model_data = TFKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.TFKT.TFKT_model import TFKT

        logger.info("Initializing TFKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0) if rc.model.use_rasch else 0

        if n_pid > 0:
            logger.info(f"TFKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.info("TFKT: Using skill-only model (Rasch disabled)")

        m = rc.model
        model = TFKT(
            n_question=metadata["num_skills"],
            n_pid=n_pid,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            final_fc_dim=m.final_fc_dim,
            n_heads=m.num_attn_heads,
            kq_same=m.kq_same,
            separate_qa=bool(m.separate_qa),
            l2=m.l2,
        )

        # Keep the source model's summed training reduction.
        loss_fn = torch.nn.BCELoss(reduction="sum")
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

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build TFKT Rasch pid data with 0 reserved for padding."""
        pid_data = question + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """TFKT 前向传播。

        TFKT 在位置 t 使用当前技能和此前交互预测 response[t]。

        Args:
            batch_data: 包含 (sequence, response, mask, question) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        y_hat_full, c_reg_loss = self.model(sequence, response, mask, pid_data)

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

        if use_pid:
            result["c_reg_loss"] = c_reg_loss

        return result

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels, question, user_id)
        """
        sequence, response, mask, late_group_id, true_labels, question, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else None

        y_hat_full, _ = self.model(sequence, response, valid_mask, pid_data)

        y_hat = torch.masked_select(y_hat_full, mask)
        y_label = torch.masked_select(true_labels, mask).float()
        group_ids = torch.masked_select(late_group_id, mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失，包含BCE损失和Rasch正则化损失"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]

        return bce_loss

    def _compute_eval_loss(self, outputs: dict) -> torch.Tensor:
        """Report average BCE, as in the source evaluation code."""
        return torch.nn.functional.binary_cross_entropy(
            outputs["y_hat"], outputs["y_label"]
        )
