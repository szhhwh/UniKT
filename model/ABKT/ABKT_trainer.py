from dataclasses import field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

from utils.config import (
    EarlyStoppingConfig,
    ModelConfig,
)
from utils.core import get_logger, register_model_config, register_trainer
from utils.model_data import create_optimized_dataloader
from utils.training import MultiTrainer, StageComponents, StageConfig

from .ABKT_data import ABKTModelData
from .ABKT_model import GMF, IRT_2, K_CMF

logger = get_logger(__name__)

__all__ = ["ABKTTrainer"]


@register_model_config("ABKT")
class ABKTConfig(ModelConfig):
    """ABKT model configuration.

    Args:
        km_hidden_dim: Hidden dimension for knowledge growth in K_CMF.
        km_guess: Guess parameter for IRT response function.
        km_lr: Learning rate for KM stage.
        km_epochs: Number of epochs for KM stage.
        km_patience: Early stopping patience for KM stage.
        am_embedding_dim: Embedding dimension for GMF.
        am_lambda: L2 regularization weight for AM.
        am_layer: Number of GNN layers in GMF (0-3).
        am_lr: Learning rate for AM stage.
        am_epochs: Number of epochs for AM stage.
        am_patience: Early stopping patience for AM stage.
        pretrain_clip: Clip value for pretrained predictions.
        combine_mode: Combine mode for KM and AM predictions: 'add' or 'mul'.
        batch_size: Batch size for AM training.
        use_adj: Whether to use learnable adjacency weights in GMF.
    """

    km_hidden_dim: int = field(
        default=5,
        metadata={"optuna": {"type": "int", "low": 3, "high": 8}},
    )
    km_guess: float = field(
        default=0.25,
        metadata={"optuna": {"type": "float", "low": 0.1, "high": 0.4}},
    )
    km_lr: float = field(
        default=0.001,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    km_epochs: int = 100
    km_patience: int = 10
    am_embedding_dim: int = field(
        default=50,
        metadata={"optuna": {"type": "int", "low": 32, "high": 128}},
    )
    am_lambda: float = 0.1
    am_layer: int = field(
        default=1,
        metadata={"optuna": {"type": "categorical", "choices": [0, 1, 2]}},
    )
    am_lr: float = field(
        default=0.0001,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    am_epochs: int = 500
    am_patience: int = 10
    pretrain_clip: float = 0.4
    combine_mode: Literal["add", "mul"] = "add"
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    use_adj: bool = True


class KMUserDataset(Data.Dataset):
    """KM 阶段的用户序列数据集（每个样本是一个用户的完整答题序列，batch_size=1）。"""

    def __init__(self, train_users: list, train_sequences: dict):
        self.train_users = train_users
        self.train_sequences = train_sequences

    def __len__(self):
        return len(self.train_users)

    def __getitem__(self, idx: int):
        user_id = self.train_users[idx]
        seq_data = self.train_sequences[user_id]
        items = torch.tensor(seq_data[0][0], dtype=torch.long)
        corrects = torch.tensor(seq_data[1][0], dtype=torch.long)
        return user_id, items, corrects


class KMValidationDataset(Data.Dataset):
    """KM 阶段的验证数据集（每个样本是一个测试三元组）。"""

    def __init__(self, test_triplets: list):
        self.test_triplets = test_triplets

    def __len__(self):
        return len(self.test_triplets)

    def __getitem__(self, idx: int):
        user_id, item_id, correct = self.test_triplets[idx]
        return user_id, item_id, correct


class AMTripletDataset(Data.Dataset):
    """AM 阶段数据集（每个样本为 [user, item, correct, km_pred, g, w]）。"""

    def __init__(self, triplets: torch.Tensor):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int):
        return self.triplets[idx]


class KMLoss(nn.Module):
    """KM 阶段损失函数：BCE Loss。"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_hat: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
        return self.bce(y_hat, y_label)


class AMLoss(nn.Module):
    """AM 阶段损失函数：Boosting Loss + L2 正则化。"""

    def __init__(self, am_lambda: float, combine_mode: str):
        super().__init__()
        self.am_lambda = am_lambda
        self.combine_mode = combine_mode

    def forward(
        self,
        pred: torch.Tensor,
        k_batch: torch.Tensor,
        g_batch: torch.Tensor,
        w_batch: torch.Tensor,
        u_norm: torch.Tensor,
        i_norm: torch.Tensor,
    ) -> torch.Tensor:
        if self.combine_mode == "add":
            loss = -torch.mean(w_batch * torch.pow(pred + g_batch / w_batch, 2))
        elif self.combine_mode == "mul":
            loss = -torch.mean(
                w_batch
                * k_batch
                * torch.pow(pred + g_batch / (w_batch * k_batch) - 1, 2)
            )
        else:
            raise ValueError(f"Unknown combine mode: {self.combine_mode}")

        return loss + self.am_lambda * (u_norm + i_norm)


@register_trainer("ABKT")
class ABKTTrainer(MultiTrainer):
    """ABKT 两阶段训练器。

    - Stage "km": 训练知识模块 K_CMF（按用户序列建模知识状态 + IRT 预测）。
    - Stage "am": 训练能力模块 GMF（图矩阵分解 + boosting 残差修正）。
    """

    def __init__(
        self,
        rc,
        data_src,
        exp_manager=None,
    ):
        """初始化 ABKT 训练器。

        Args:
            rc: RunConfig (OmegaConf DictConfig)。
            data_src: 数据源。
            exp_manager: 实验管理器。
        """
        logger.info("Preparing ABKT data...")
        self.model_data = ABKTModelData(data_src)
        self.data = self.model_data.prepare_data(rc)

        # Cross-stage state, populated by stage builders / on_stage_complete
        self.km_model: K_CMF | None = None
        self.am_model: GMF | None = None
        self.Q_matrix: torch.Tensor | None = None
        self.am_train_triplets: torch.Tensor | None = None
        self.am_test_triplets: torch.Tensor | None = None
        self.adj_norm: torch.Tensor | None = None

        super().__init__(rc, data_src, exp_manager)

    def build_stages(self) -> list[StageConfig]:
        """声明两个训练阶段。构建器延迟执行，am 可依赖 km 的输出。"""
        return [
            StageConfig(name="km", build=self._build_km),
            StageConfig(name="am", build=self._build_am),
        ]

    def _build_km(self) -> StageComponents:
        """构建 KM 阶段：K_CMF 模型 + 单用户序列数据 + BCE 损失。"""
        m = self.run_config.model
        Q_matrix = self.data["Q_matrix"].to(self.device_)
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]
        num_skills = self.data["num_skills"]

        self.km_model = K_CMF(
            k_hidden_size=m.km_hidden_dim,
            skill_num=num_skills,
            user_num=num_users,
            item_num=num_items,
            q_matrix=Q_matrix,
        )
        self.Q_matrix = Q_matrix

        train_dataset = KMUserDataset(
            train_users=self.data["train_users"],
            train_sequences=self.data["train_sequences"],
        )
        train_loader = create_optimized_dataloader(
            train_dataset, batch_size=1, shuffle=True, device=self.device_
        )

        val_dataset = KMValidationDataset(self.data["test_triplets"])
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=len(val_dataset), shuffle=False, device=self.device_
        )

        optimizer = optim.Adam(self.km_model.parameters(), lr=m.km_lr)

        logger.info(
            f"KM Stage: users={num_users}, items={num_items}, skills={num_skills}"
        )

        return StageComponents(
            model=self.km_model,
            optimizer=optimizer,
            loss_fn=KMLoss(),
            train_data=train_loader,
            val_data=val_loader,
            epochs=m.km_epochs,
            early_stopping=EarlyStoppingConfig(
                monitor="acc", mode="max", patience=m.km_patience
            ),
            # Early-stop on ACC but checkpoint/hand back the best-AUC model
            checkpoint_monitor="auc",
            checkpoint_mode="max",
        )

    def _build_am(self) -> StageComponents:
        """构建 AM 阶段：GMF 模型 + boosting 残差三元组 + boosting 损失。"""
        m = self.run_config.model
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]

        if self.am_train_triplets is None:
            raise RuntimeError(
                "AM stage requires boosting residuals. "
                "Ensure the KM stage has completed (on_stage_complete)."
            )

        self.am_model = GMF(
            n_users=num_users,
            n_items=num_items,
            embedding_k=m.am_embedding_dim,
            aj_norm=self.adj_norm,
            adj=m.use_adj,
            layer=m.am_layer,
        )

        train_dataset = AMTripletDataset(self.am_train_triplets)
        train_loader = create_optimized_dataloader(
            train_dataset,
            batch_size=m.batch_size,
            shuffle=True,
            device=self.device_,
        )

        val_dataset = AMTripletDataset(self.am_test_triplets)
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=len(val_dataset), shuffle=False, device=self.device_
        )

        optimizer = optim.Adam(self.am_model.parameters(), lr=m.am_lr)
        loss_fn = AMLoss(am_lambda=m.am_lambda, combine_mode=m.combine_mode)

        logger.info(
            f"AM Stage: train_triplets={len(self.am_train_triplets)}, "
            f"test_triplets={len(self.am_test_triplets)}"
        )

        return StageComponents(
            model=self.am_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_loader,
            val_data=val_loader,
            epochs=m.am_epochs,
            early_stopping=EarlyStoppingConfig(
                monitor="acc", mode="max", patience=m.am_patience
            ),
            # Early-stop on ACC but report/checkpoint the best-AUC model
            checkpoint_monitor="auc",
            checkpoint_mode="max",
        )

    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """模型前向传播，按当前阶段分发。"""
        if self._current_stage == "km":
            return self._forward_km(batch_data)
        if self._current_stage == "am":
            return self._forward_am(batch_data)
        raise ValueError(f"Unknown stage: {self._current_stage!r}")

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失，按当前阶段分发。"""
        if self._current_stage == "km":
            return self.loss(outputs["y_hat"], outputs["y_label"])

        # Validation has no g/w, so fall back to BCE on the combined prediction
        if "g" not in outputs or "w" not in outputs:
            y_hat = outputs["y_hat"].clamp(1e-6, 1 - 1e-6)
            return F.binary_cross_entropy(y_hat, outputs["y_label"])
        return self.loss(
            outputs["am_pred"],
            outputs["km_pred"],
            outputs["g"],
            outputs["w"],
            outputs["u_norm"],
            outputs["i_norm"],
        )

    def on_stage_complete(self, name: str, result) -> None:
        """KM 阶段结束后计算 boosting 残差，供 AM 阶段使用。"""
        if name == "km":
            logger.info("Computing boosting residuals for AM stage...")
            self._compute_boosting_residuals()

    def _forward_km(self, batch_data: tuple) -> dict:
        """KM 阶段的前向传播。

        训练时 batch_size=1，每次处理一个用户的完整序列；
        验证时 batch 包含所有测试三元组 (user_id, item_id, correct)。
        """
        user_ids, items, corrects = batch_data

        # Dispatch on shape: train feeds [1, seq_len], val feeds [batch_size]
        if items.dim() == 2:
            return self._forward_km_train(user_ids, items, corrects)
        return self._forward_km_val(user_ids, items, corrects)

    def _forward_km_train(
        self, user_ids: torch.Tensor, items: torch.Tensor, corrects: torch.Tensor
    ) -> dict:
        """KM 阶段训练时的前向传播。"""
        # Train loader uses batch_size=1; drop the singleton batch dim
        user_id = user_ids.item()
        items = items.squeeze(0).to(self.device_)
        corrects = corrects.squeeze(0).to(self.device_)

        user_k, _, _ = self.km_model(user_id, items)
        item_q = self.Q_matrix[items, :]
        item_k = self.km_model.item_k[items, :]

        pred = IRT_2(
            user_k[:-1, :], item_k, item_q, self.run_config.model.km_guess
        ).clamp(1e-6, 1 - 1e-6)

        return {
            "y_hat": pred,
            "y_label": corrects.float(),
            "y_predict": (pred >= 0.5).int(),
            "y_score": pred,
            "y_prob": pred,
        }

    def _forward_km_val(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor, corrects: torch.Tensor
    ) -> dict:
        """KM 阶段验证时的前向传播（批量处理模式）。

        先批量计算所有测试用户的最终知识状态，再批量预测所有测试样本。

        Args:
            user_ids: 用户 ID [batch_size]
            item_ids: 测试题目 ID [batch_size]
            corrects: 实际正确率 [batch_size]
        """
        user_ids = user_ids.to(self.device_)
        item_ids = item_ids.to(self.device_)
        corrects = corrects.to(self.device_)

        test_user_state_dict = {}
        with torch.no_grad():
            for test_user_id in self.data["test_users"]:
                if test_user_id in self.data["train_sequences"]:
                    train_items = self.data["train_sequences"][test_user_id][0][0]
                    train_items_tensor = torch.tensor(
                        train_items, dtype=torch.long, device=self.device_
                    )
                    user_k, _, _ = self.km_model(test_user_id, train_items_tensor)
                    test_user_state_dict[test_user_id] = user_k[-1, :]
                else:
                    # Users without training history fall back to the initial knowledge state
                    test_user_state_dict[test_user_id] = torch.sigmoid(
                        self.km_model.user_initial_k[test_user_id, :]
                    )

        user_states_k = torch.cat(
            [
                test_user_state_dict[user_id].unsqueeze(0)
                for user_id in user_ids.tolist()
            ],
            dim=0,
        )  # [batch_size, skill_num]

        item_states_q = self.Q_matrix[item_ids, :]  # [batch_size, skill_num]
        item_state_k = self.km_model.item_k[item_ids, :]  # [batch_size, skill_num]

        pred = IRT_2(
            user_states_k, item_state_k, item_states_q, self.run_config.model.km_guess
        ).clamp(1e-6, 1 - 1e-6)

        return {
            "y_hat": pred,
            "y_label": corrects.float(),
            "y_predict": (pred >= 0.5).int(),
            "y_score": pred,
            "y_prob": pred,
        }

    def _forward_am(self, batch_data: tuple) -> dict:
        """AM 阶段的前向传播。

        训练数据: [user, item, correct, km_pred, g, w] - 6 列
        验证数据: [user, item, correct, km_pred] - 4 列
        """
        # batch: [batch, 6] 或 [batch, 4]
        batch = batch_data[0] if isinstance(batch_data, tuple) else batch_data
        batch = batch.to(self.device_)

        u_idx = batch[:, 0].long()
        i_idx = batch[:, 1].long()
        correct = batch[:, 2]
        km_pred = batch[:, 3]

        am_pred, u_norm, i_norm = self.am_model(u_idx, i_idx)

        if self.run_config.model.combine_mode == "mul":
            final_pred = am_pred * km_pred
        else:
            final_pred = am_pred + km_pred

        output = {
            "y_hat": final_pred,
            "y_label": correct,
            "y_predict": (final_pred >= 0.5).int(),
            "y_score": final_pred,
            "y_prob": final_pred.clamp(0.0, 1.0),
            "am_pred": am_pred,
            "km_pred": km_pred,
            "u_norm": u_norm,
            "i_norm": i_norm,
        }

        # Training rows carry g/w columns for the boosting loss
        if batch.shape[1] >= 6:
            output["g"] = batch[:, 4]
            output["w"] = batch[:, 5]

        return output

    def _compute_boosting_residuals(self):
        """基于训练好的 KM 模型计算 boosting 残差，并构建 AM 阶段所需数据。"""
        m = self.run_config.model
        Q_matrix = self.Q_matrix
        train_sequences = self.data["train_sequences"]
        test_triplets = self.data["test_triplets"]
        train_users = self.data["train_users"]
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]

        self.km_model.eval()

        train_itemsq = []
        train_correctsq = []
        for user_id in train_users:
            seq_data = train_sequences[user_id]
            train_itemsq.append(torch.tensor(seq_data[0][0], dtype=torch.long))
            train_correctsq.append(torch.tensor(seq_data[1][0], dtype=torch.long))

        train_triplet_list = []
        user_final_states = {}

        with torch.no_grad():
            for idx, user_id in enumerate(
                tqdm(train_users, desc="Computing KM predictions")
            ):
                itemsq = train_itemsq[idx].to(self.device_)
                correctsq = train_correctsq[idx].to(self.device_)

                user_k, _, _ = self.km_model(user_id, itemsq)
                item_q = Q_matrix[itemsq, :]
                item_k = self.km_model.item_k[itemsq, :]

                clip_pred = IRT_2(user_k[:-1, :], item_k, item_q, m.km_guess).clamp(
                    m.pretrain_clip, 1 - m.pretrain_clip
                )

                user_final_states[user_id] = user_k[-1, :]

                for i in range(len(itemsq)):
                    train_triplet_list.append(
                        [
                            user_id,
                            itemsq[i].item(),
                            correctsq[i].item(),
                            clip_pred[i].item(),
                        ]
                    )

        test_triplet_list = []
        with torch.no_grad():
            for user_id, item_id, correct in tqdm(
                test_triplets, desc="Computing test KM predictions"
            ):
                if user_id in user_final_states:
                    user_k = user_final_states[user_id]
                    item_q = Q_matrix[item_id, :]
                    item_k = self.km_model.item_k[item_id, :]
                    clip_pred = IRT_2(
                        user_k.unsqueeze(0),
                        item_k.unsqueeze(0),
                        item_q.unsqueeze(0),
                        m.km_guess,
                    ).clamp(m.pretrain_clip, 1 - m.pretrain_clip)
                    test_triplet_list.append(
                        [user_id, item_id, correct, clip_pred.item()]
                    )

        train_triplets = torch.tensor(train_triplet_list, dtype=torch.float32)
        self.am_test_triplets = torch.tensor(test_triplet_list, dtype=torch.float32)

        # boosting 残差: g = y/k - (1-y)/(1-k), w = -(y/k^2 + (1-y)/(1-k)^2)
        y = train_triplets[:, 2]
        k = train_triplets[:, 3]
        g = y / k - (1 - y) / (1 - k)
        w = -(y / (k**2) + (1 - y) / ((1 - k) ** 2))

        self.am_train_triplets = torch.cat(
            [train_triplets, g.unsqueeze(1), w.unsqueeze(1)], dim=1
        )

        logger.info(f"Train triplets shape: {self.am_train_triplets.shape}")
        logger.info(f"Test triplets shape: {self.am_test_triplets.shape}")

        self.adj_norm = self.model_data.build_normalized_adj_matrix(
            train_sequences=train_sequences,
            num_users=num_users,
            num_items=num_items,
            symmetric=True,
        ).to(self.device_)
