"""TCKT 模型训练器。

负责：
    1. 数据准备（委托 :class:`TCKTModelData`）与模型构建；
    2. ``forward_pass``：调用模型并按 same-position 约定提取有效预测；
    3. **在线全局字典生成**：通过回调在每个 epoch 开始时，对当前模型在训练集上产出的
       交互嵌入做 MiniBatchKMeans 聚类，得到 N 个中心，更新模型的全局字典缓冲区
       （论文 4.3 节“dictionary is updated during the end-to-end training”）。

训练基础设施（``KTM`` 类）由 ``BaseTrainer`` 提供，全局聚类生成由此处的
:class:`GlobalDictRefreshCallback` + :meth:`refresh_global_dict` 实现。
"""

from __future__ import annotations

from dataclasses import field

import numpy as np
import torch
from torch import nn

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents
from utils.training.callbacks import Callback

logger = get_logger(__name__)


@register_model_config("TCKT")
class TCKTConfig(ModelConfig):
    """TCKT model configuration.

    Args:
        d_k: Hidden dimension d_k.
        d_a: Expansion dim for answer / difficulty (paper d_a).
        d_e: Exercise embedding dim.
        num_heads: Attention heads.
        dropout: Dropout rate.
        q_gamma: Smoothing added to each event's skill indicator.
        global_dict_size: Global dictionary size N (K-Means clusters).
        max_rt_seconds: Response time cap in seconds (n_at = this + 1).
        max_it_minutes: Interval time cap in minutes (n_it = this + 1).
        cluster_sample_size: Max interactions sampled for K-Means each epoch.
        learning_rate: Learning rate.
        adam_beta1: Adam beta1 (reference code uses 0.1).
        weight_decay: Weight decay.
        max_grad_norm: Max gradient norm for clipping, 0 disables.
        lr_decay_step: StepLR step size.
        lr_decay_rate: StepLR gamma.
        epochs: Number of training epochs.
        batch_size: Batch size.
    """

    # powers of two so d_k % num_heads == 0 for every combination
    d_k: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    d_a: int = 64
    d_e: int = 128
    num_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    q_gamma: float = 0.03
    global_dict_size: int = 400
    max_rt_seconds: int = 300
    max_it_minutes: int = 120
    cluster_sample_size: int = 300000
    learning_rate: float = field(
        default=0.002,
        metadata={
            "optuna": {"type": "float", "low": 0.0005, "high": 0.005, "log": True}
        },
    )
    adam_beta1: float = 0.1
    weight_decay: float = field(
        default=1e-6,
        metadata={"optuna": {"type": "float", "low": 1e-7, "high": 1e-3, "log": True}},
    )
    max_grad_norm: float = 1.0
    lr_decay_step: int = 10
    lr_decay_rate: float = 0.5
    epochs: int = 100
    batch_size: int = field(
        default=32,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


class GlobalDictRefreshCallback(Callback):
    """每个 epoch 开始时刷新全局字典。

    - epoch 0：用随机初始化的模型产出初始字典（冷启动）。
    - 后续 epoch：用上一轮更新后的模型产出新字典。
    这样同一 epoch 的训练与验证使用同一份字典。
    """

    def __init__(self, trainer: TCKTTrainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch: int, **kwargs):
        self.trainer.refresh_global_dict()


@register_trainer("TCKT")
class TCKTTrainer(BaseTrainer):
    """TCKT 训练器。"""

    def build_components(self, rc, data_src):
        # Force the math SDP backend for nn.MultiheadAttention. PyTorch >=2.x
        # defaults to flash / memory-efficient attention, whose backward kernel
        # returns NaN gradients once this model's attention becomes peaked (which
        # happens as weights grow on large datasets such as assistments12). The NaN
        # gradient silently corrupts the embedding weights, and the next forward
        # feeds NaN predictions into BCELoss -> device-side assert. The math backend
        # computes the same attention with a numerically robust backward.
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        from model.TCKT.TCKT_data import TCKTModelData
        from model.TCKT.TCKT_model import TCKTNet

        train_dataset, val_dataset, test_dataset, info = TCKTModelData(
            data_src
        ).prepare_data(rc)
        m = rc.model
        model = TCKTNet(
            num_questions=info["num_questions"],
            num_skills=info["num_skills"],
            n_at=info["n_at"],
            n_it=info["n_it"],
            d_k=m.d_k,
            d_a=m.d_a,
            d_e=m.d_e,
            num_heads=m.num_heads,
            seq_len=info["max_question_seq_len"],
            global_dict_size=m.global_dict_size,
            dropout=m.dropout,
            q_gamma=m.q_gamma,
        )
        model.set_difficulty(info["difficulty"])

        # Clustering config used by the refresh callback.
        self.global_dict_size = m.global_dict_size
        self.cluster_sample_size = m.cluster_sample_size
        # refresh_global_dict seeds its K-Means clustering from this attribute.
        self.seed = rc.general.seed

        loss_fn = nn.BCELoss(reduction="none")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            eps=1e-8,
            betas=(m.adam_beta1, 0.999),
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

    def build_callbacks(self):
        return [GlobalDictRefreshCallback(self)]

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """TCKT 前向传播。

        模型输出 ``pred[:, t]`` 预测 ``a[:, t]``（same-position 约定）。
        """
        e, at, a, it, c, mask, skills = batch_data
        e = self._move_tensor_to_device(e)
        at = self._move_tensor_to_device(at)
        a = self._move_tensor_to_device(a)
        it = self._move_tensor_to_device(it)
        c = self._move_tensor_to_device(c)
        mask = self._move_tensor_to_device(mask)
        skills = self._move_tensor_to_device(skills)

        pred = self.model(e, at, a, it, c, skills)  # [B, S]

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

    @torch.inference_mode()
    def refresh_global_dict(self) -> None:
        """对当前模型在训练集上的交互嵌入做 K-Means，更新全局字典。"""
        from sklearn.cluster import MiniBatchKMeans

        device = self.device_
        rng = np.random.RandomState(self.seed)
        cap = self.cluster_sample_size

        chunks: list[np.ndarray] = []
        collected = 0
        for batch in self.train_data:
            e, _at, a, _it, c, mask, _skills = batch
            e = e.to(device)
            c = c.to(device)
            a = a.to(device)
            mask = mask.to(device)
            x = self.model.compute_interaction_embeddings(e, c, a)  # [B, S, d_k]
            emb = x[mask].detach().cpu().numpy().astype(np.float32)
            chunks.append(emb)
            collected += emb.shape[0]
            # Cap peak memory: once we accumulate 2x the cap, sample-merge back down.
            if collected >= cap * 2:
                merged = np.concatenate(chunks, axis=0)
                idx = rng.choice(merged.shape[0], cap, replace=False)
                chunks = [merged[idx]]
                collected = cap

        embeds = np.concatenate(chunks, axis=0)
        if embeds.shape[0] > cap:
            idx = rng.choice(embeds.shape[0], cap, replace=False)
            embeds = embeds[idx]

        # If fewer samples than clusters, upsample with replacement so K-Means yields N centers.
        n_samples = embeds.shape[0]
        if n_samples < self.global_dict_size:
            idx = rng.choice(n_samples, self.global_dict_size, replace=True)
            embeds = embeds[idx]

        km = MiniBatchKMeans(
            n_clusters=self.global_dict_size,
            random_state=int(self.seed),
            n_init=3,
            batch_size=4096,
        )
        km.fit(embeds)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        self.model.update_global_dict(centers)
        logger.debug(
            f"Global dict refreshed: clustered {embeds.shape[0]} embeddings "
            f"into {self.global_dict_size} centers"
        )
