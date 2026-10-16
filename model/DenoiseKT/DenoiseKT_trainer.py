"""DenoiseKT 模型训练器。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DenoiseKT")
class DenoiseKTConfig(ModelConfig):
    """DenoiseKT 模型配置.

    Args:
        d_model: Hidden dimension of the model.
        n_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        dropout: Dropout probability for the transformer.
        dropout1: Dropout probability for the GCN.
        d_ff: Feed-forward network dimension.
        final_fc_dim: First output MLP dimension.
        final_fc_dim2: Second output MLP dimension.
        bf: Distance-decay base for same-concept boost focus.
        kq_same: Whether key and query share the linear projection (1=yes, 0=no).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
    """

    # powers of two so d_model % num_attn_heads == 0 for every combination
    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256, 512]}},
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
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    dropout1: float = 0.1
    d_ff: int = 64
    final_fc_dim: int = 256
    final_fc_dim2: int = 256
    bf: float = 0.9
    kq_same: int = 1
    epochs: int = 200
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}},
    )
    lr_decay: float | None = None
    # linear range: log sampling requires low > 0, default is 0.0
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 1e-2}},
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128, 256]}},
    )


@register_trainer("DenoiseKT")
class DenoiseKTTrainer(BaseTrainer):
    """DenoiseKT 模型训练器。"""

    def build_components(self, rc, data_src):
        from model.DenoiseKT.DenoiseKT_data import DenoiseKTModelData

        model_data = DenoiseKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            question_graph,
        ) = model_data.prepare_data(rc)

        from model.DenoiseKT.DenoiseKT_model import DenoiseKT

        metadata = data_src.get_metadata()
        num_q = metadata["num_questions"]
        num_c = metadata["num_skills"]
        logger.info("Initializing DenoiseKT model...")

        m = rc.model
        model = DenoiseKT(
            num_c=num_c,
            num_q=num_q,
            question_graph=question_graph,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            dropout1=m.dropout1,
            bf=m.bf,
            d_ff=m.d_ff,
            seq_len=metadata["max_question_seq_len"],
            kq_same=m.kq_same,
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            num_attn_heads=m.num_attn_heads,
        )

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
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict:
        """前向传播。

        预测语义：
        ``preds[:, t]`` 使用 ``question[:, t]`` 与历史 ``qa[:, :t]`` 预测 ``response[:, t]``。

        batch_data: ``(question, response, mask, skills)``
        """
        question, response, mask, skills = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        skills = self._move_tensor_to_device(skills)

        preds = self.model(question, response, skills)  # [B, S]

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
