# 自定义回调与损失函数

UniKT 提供了灵活且非侵入式的训练控制机制：
1. **回调系统**：允许在训练生命周期的不同阶段注入自定义操作，例如 Epoch 起止、Phase 切换以及 Batch 迭代时的特征刷新或内存回收。
2. **损失解耦机制**：支持在训练时引入辅助任务损失、对比学习损失或正则化项，同时严格确保验证集与测试集评估不受辅助损失干扰。

---

## 回调系统生命周期

UniKT 的 `Callback` 运行于 `BaseTrainer` 与 `MultiTrainer` 的训练主循环中。各钩子方法的触发时机由以下训练循环流程决定：

```python
# 训练主循环与回调触发时机
def fit(epochs):
    cb.on_train_begin(epochs)
    try:
        for epoch in range(epochs):
            cb.on_epoch_begin(epoch)
            for phase in ["train", "val"]:
                cb.on_phase_begin(epoch, phase)
                for batch_idx, batch in enumerate(dataloader):
                    cb.on_batch_begin(epoch, batch_idx, phase)
                    loss, metrics = forward_and_backward(batch)
                    cb.on_batch_end(epoch, batch_idx, phase, loss)
                cb.on_phase_end(epoch, phase, phase_loss, metrics)
            cb.on_epoch_end(epoch, train_loss, val_loss)
            if cb.should_stop():
                break
        cb.on_train_end()
    finally:
        cb.close()
```

### 回调钩子接口全览

所有自定义回调均继承自 `utils.training.Callback`，可按需重写以下方法：

```python
from typing import Any
from utils.training import Callback


class MyCallback(Callback):
    def on_train_begin(self, epochs: int, **kwargs: Any) -> None:
        """在整个训练开始前调用一次。"""
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        """在训练正常完成时调用；若中途异常退出则不触发，使用 close 代替。"""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """在每个 epoch 迭代开始前调用。"""
        pass

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float | None, **kwargs: Any
    ) -> None:
        """在每个 epoch 结束且完成 train 和 val 后调用。"""
        pass

    def on_phase_begin(self, epoch: int, phase: str, **kwargs: Any) -> None:
        """在进入 train 或 val 阶段时调用。"""
        pass

    def on_phase_end(
        self,
        epoch: int,
        phase: str,
        loss: float,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> None:
        """在完成某个 phase 的全部 batch 评估与指标累积后调用。"""
        pass

    def on_batch_begin(
        self, epoch: int, batch_idx: int, phase: str, **kwargs: Any
    ) -> None:
        """在批次前向传播开始前调用。"""
        pass

    def on_batch_end(
        self,
        epoch: int,
        batch_idx: int,
        phase: str,
        loss: float,
        **kwargs: Any,
    ) -> None:
        """在批次前向传播与反向传播完成后调用。"""
        pass

    def should_stop(self, **kwargs: Any) -> bool:
        """轮询是否需要提前终止训练，返回 True 则立即触发早停退出。"""
        return False

    def close(self) -> None:
        """在 Trainer 的 finally 保护块中调用，无论正常完成还是报错均会执行，用于释放资源。"""
        pass
```

---

## 实战：编写与注册自定义回调

在模型训练器（`<Name>_trainer.py`）中，重写 `build_callbacks()` 方法，返回自定义回调实例的列表即可完成装配。

### 案例 1：周期性动态特征聚类与字典更新

在 TCKT 模型中，每轮训练前需要用上一轮学到的模型交互嵌入在训练集上进行 Mini-Batch K-Means 聚类，动态更新一个全局特征字典。

```python
from utils.training import BaseTrainer, Callback


class GlobalDictRefreshCallback(Callback):
    """TCKT 全局字典周期性更新回调。"""

    def __init__(self, trainer: "TCKTTrainer"):
        self.trainer = trainer

    def on_epoch_begin(self, epoch: int, **kwargs):
        """每个 epoch 开始时重新聚类并更新模型字典。"""
        self.trainer.refresh_global_dict()


class TCKTTrainer(BaseTrainer):
    def build_callbacks(self) -> list[Callback]:
        """向框架注册 TCKT 特有的回调。"""
        return [GlobalDictRefreshCallback(self)]

    @torch.inference_mode()
    def refresh_global_dict(self) -> None:
        """聚类逻辑实现。"""
        self.model.eval()
        # 遍历训练集提取潜变量并执行 K-Means...
        # self.model.global_dict.data.copy_(...)
```

### 案例 2：显存清理与温度调度回调

```python
import torch
from utils.training import Callback


class AnnealingAndCleanupCallback(Callback):
    """动态调整退火温度并在 Epoch 结束清理 CUDA 缓存。"""

    def __init__(self, trainer, initial_temp: float = 1.0, decay: float = 0.95):
        self.trainer = trainer
        self.temp = initial_temp
        self.decay = decay

    def on_epoch_begin(self, epoch: int, **kwargs):
        self.temp *= self.decay
        # 更新模型中的温度参数
        if hasattr(self.trainer.model, "temperature"):
            self.trainer.model.temperature = self.temp

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float | None, **kwargs
    ):
        # 显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## 自定义损失函数机制

UniKT 在架构上清晰解耦了**训练损失**与**评估损失**。

### 解耦设计规范

```python
# BaseTrainer 中的默认实现
def _compute_loss(self, outputs: dict) -> torch.Tensor:
    """仅在训练阶段调用，用于 loss.backward()。"""
    return self.loss(outputs["y_hat"], outputs["y_label"])


def _compute_eval_loss(self, outputs: dict) -> torch.Tensor:
    """在验证与测试阶段调用，仅用于统计评估损失。"""
    return self.loss(outputs["y_hat"], outputs["y_label"])
```

- **`_compute_loss`**：用于反向传播梯度。模型可以在此处加入**对比学习损失**、**图正则化损失**、**知识状态平滑项**或**多任务辅助损失**。
- **`_compute_eval_loss`**：默认只使用纯净的主预测损失（如标准二元交叉熵 BCELoss），确保在验证集和测试集上记录的 `loss` 指标在不同模型间具备公平可比性，不被辅助项人为拉大。

### 案例：接入对比学习辅助损失

在 CL4KT 中，除了主答题预测损失外，还需要计算原序列与数据增强序列之间的对比损失：

```python
class CL4KTTrainer(BaseTrainer):
    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        # 1. 前向传播计算主任务预测与对比损失
        y_hat, cl_loss = self.model(batch_data)

        y_hat_valid, y_label_valid, _ = self._extract_valid_predictions(
            y_hat, batch_data.response, batch_data.mask
        )

        return {
            "y_hat": y_hat_valid,
            "y_label": y_label_valid,
            "y_predict": self._generate_binary_predictions(y_hat_valid),
            "cl_loss": cl_loss,  # 将辅助损失放入输出字典中
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """重写训练损失计算：主 BCE 损失 + 对比损失加权。"""
        # 主任务二元交叉熵损失
        bce_loss = self.loss(outputs["y_hat"], outputs["y_label"])

        # 从 RunConfig 读取对比损失超参数权重
        reg_weight = self.run_config.model.reg_cl

        if "cl_loss" in outputs and reg_weight > 0:
            return bce_loss + reg_weight * outputs["cl_loss"]

        return bce_loss
```

此时：
- 训练反向传播的是：`bce_loss + reg_weight * cl_loss`；
- 验证集和测试集日志记录的则是标准的纯净 `bce_loss`，既保证了训练时多任务辅助学习生效，又确保了下游模型评测指标不受干扰。
