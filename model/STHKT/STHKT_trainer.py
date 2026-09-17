"""STHKT 模型训练器"""

from dataclasses import field
from typing import Literal

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


class NoamOpt:
    """内嵌 Noam 学习率调度的优化器包装。

    ``lr(step) = factor * model_size^-0.5 * min(step^-0.5, step * warmup^-1.5)``，
    按训练 step（batch）推进。
    """

    def __init__(self, model_size: int, factor: int, warmup: int, optimizer):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0

    def rate(self, step: int | None = None) -> float:
        step = self._step if step is None else step
        return self.factor * (
            self.model_size**-0.5 * min(step**-0.5, step * self.warmup**-1.5)
        )

    def step(self):
        self._step += 1
        rate = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"noam_step": self._step, "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        self._step = state.get("noam_step", 0)
        self.optimizer.load_state_dict(state["optimizer"])

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    @property
    def defaults(self):
        return self.optimizer.defaults


@register_model_config("STHKT")
class STHKTConfig(ModelConfig):
    """STHKT model configuration.

    Args:
        d_model: Hidden dimension of the model.
        num_heads: Number of attention heads.
        d_ff: Feed-forward network dimension.
        dropout: Dropout probability.
        result_hidden_dim: Hidden dimension of the prediction layer.
        graph_mode: Structural module ("gcn" or "random_walk").
        hawkes_mode: Hawkes attention variant (paper Table 4 ablations).
        graph_scatter: How repeated questions inside one sequence aggregate
            when scattered onto graph nodes ("mean" or "sum").
        noam_factor: Noam schedule factor.
        noam_warmup: Noam schedule warmup steps.
        epochs: Number of training epochs.
        learning_rate: Initial Adam learning rate; overwritten by the Noam
            schedule from the first step on and kept for reference only.
        batch_size: Batch size for training.
    """

    # One of the three fixed parameters in the paper's experiments
    # (batch / d_model / seq_len).
    d_model: int = 128
    num_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8]}},
    )
    d_ff: int = 2048
    dropout: float = field(
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    result_hidden_dim: int = 256
    graph_mode: Literal["gcn", "random_walk"] = "gcn"
    hawkes_mode: Literal["full", "without_hawkes", "nn", "qkc", "qs", "none"] = "full"
    graph_scatter: Literal["mean", "sum"] = "mean"
    noam_factor: int = 1
    noam_warmup: int = 4000
    epochs: int = 50
    learning_rate: float = 0.1
    batch_size: int = 50


@register_trainer("STHKT")
class STHKTTrainer(BaseTrainer):
    """STHKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.STHKT.STHKT_data import STHKTModelData

        model_data = STHKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, info = model_data.prepare_data(rc)

        from model.STHKT.STHKT_model import STHKT

        logger.info("Initializing STHKT model...")
        m = rc.model
        model = STHKT(
            num_questions=info["num_questions"],
            num_skills=info["num_skills"],
            num_users=info["num_users"],
            max_seq_len=info["max_seq_len"],
            d_model=m.d_model,
            num_heads=m.num_heads,
            d_ff=m.d_ff,
            dropout=m.dropout,
            result_hidden_dim=m.result_hidden_dim,
            graph_mode=m.graph_mode,
            hawkes_mode=m.hawkes_mode,
            graph_scatter=m.graph_scatter,
            graph_data=info["graph_data"],
        )

        optimizer = NoamOpt(
            m.d_model,
            m.noam_factor,
            m.noam_warmup,
            torch.optim.Adam(
                model.parameters(),
                lr=m.learning_rate,
                weight_decay=m.weight_decay,
            ),
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            lr_scheduler=None,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """STHKT 前向传播。

        预测语义：
        - y_hat[:, t] 使用 sequence[0:t] 与 response[0:t] 以及当前题目
          embedding[0:t+1] 预测 response[t]（same-position 语义）
        - 框架统一的 next-item 对齐会丢弃每个序列窗口的首个预测位
          （t=0 无历史），损失/指标因此不含位置 0 的样本

        Args:
            batch_data: (question, response, mask) 元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        question, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(question, response).squeeze(-1)  # [B, S]

        # Padded responses (value 2) exist only where mask is False.
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
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
