"""HDHKT 模型训练器。

定义 HDHKT 模型特定的训练逻辑。
"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("HDHKT")
class HDHKTConfig(ModelConfig):
    """HDHKT 模型配置。

    Args:
        hidden_dim: Hidden layer dimension.
        n_hop: Number of GNN hops.
        heads: Number of attention heads.
        lstm_layers: Number of LSTM layers.
        history_neighbour: History neighbor count.
        att_bound: Attention bound.
        num_difficulty_clusters: Number of difficulty clusters for weighted hypergraph.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        dropout: Dropout rate.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    hidden_dim: int = field(
        default=250,
        metadata={"optuna": {"type": "int", "low": 128, "high": 512}},
    )
    n_hop: int = field(
        default=4,
        metadata={"optuna": {"type": "int", "low": 2, "high": 6}},
    )
    heads: int = 1
    lstm_layers: int = 1
    history_neighbour: int = 5
    att_bound: float = 0.1
    num_difficulty_clusters: int = 5
    epochs: int = 120
    learning_rate: float = field(
        default=0.0003,
        metadata={
            "optuna": {"type": "float", "low": 0.00001, "high": 0.001, "log": True}
        },
    )
    lr_decay: float | None = None
    dropout: float = field(
        default=0.25,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    weight_decay: float = field(
        default=0.00001,
        metadata={
            "optuna": {"type": "float", "low": 1e-06, "high": 0.0001, "log": True}
        },
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("HDHKT")
class HDHKTTrainer(BaseTrainer):
    """HDHKT 模型训练器。

    负责初始化 HDHKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.HDHKT.HDHKT_data import HDHKTModelData

        model_data = HDHKTModelData(data_src)
        data_dict = model_data.prepare_data(rc)

        train_dataset = data_dict["train_dataset"]
        val_dataset = data_dict["val_dataset"]
        test_dataset = data_dict.get("test_dataset")
        self.hypergraph = data_dict["skill_hypergraph"]
        self.hetero_graph = data_dict["hetero_graph"]

        from model.HDHKT.HDHKT_model import HDHKT

        logger.info("Initializing HDHKT model...")
        m = rc.model
        model = HDHKT(
            data_metadata=data_src.get_metadata(),
            hetero_metadata=self.hetero_graph.metadata(),
            hidden_dim=m.hidden_dim,
            n_hop=m.n_hop,
            heads=m.heads,
            lstm_layers=m.lstm_layers,
            dropout=m.dropout,
            history_neighbour=m.history_neighbour,
            att_bound=m.att_bound,
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        device = (
            torch.device(rc.general.device) if rc.general.device else self._try_gpu()
        )
        self.hetero_graph = self.hetero_graph.to(device)
        self.hypergraph = self.hypergraph.to(device)
        _ = self.hypergraph.L_HGNN

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
    ) -> dict[str, torch.Tensor]:
        """HDHKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成。

        Args:
            batch_data: (sequence, response, mask, skills)

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, skills = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        skills = self._move_tensor_to_device(skills)
        mask = self._move_tensor_to_device(mask)

        # Model output at step t predicts the label at step t+1
        y_hat_full = self.model(
            sequence,
            response,
            mask,
            self.hetero_graph,
            self.hypergraph,
            skills,
        )  # [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
