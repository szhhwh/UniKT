# 多阶段训练架构

部分先进的知识追踪模型采用了复合、分步或级联提升的训练范式。例如 ABKT 首先训练知识状态模块 KM 阶段，利用 IRT 拟合学生对知识点的掌握度；待 KM 阶段收敛并冻结后，计算作答残差与样本权重，进而进入能力模块 AM 阶段训练残差网络，最后由两阶段联合完成最终推断。

针对复杂的阶段化训练流程，UniKT 提供了 `MultiTrainer` 训练架构。其执行时序包含四个关键环节：

1. **获取阶段编排**：`MultiTrainer.run()` 触发 `build_stages()` 获取阶段配置列表。
2. **延迟构建与独立训练**：按顺序逐个阶段调用构建函数，每个阶段独立拥有专用的模型、优化器、早停监控与学习率调度器。
3. **阶段流转与残差通信**：阶段收敛完成后，触发 `on_stage_complete(stage_name, result)` 钩子，计算中间残差或更新下一阶段的输入特征。
4. **测试集联合推断**：所有阶段训练完成后触发测试评估，调用 `test_forward_pass(batch_data)` 联合各阶段模型完成最终推断。

---

## 核心架构设计

单阶段训练器 `BaseTrainer` 与多阶段训练器 `MultiTrainer` 的架构差异如下：

| 对比维度 | BaseTrainer 单阶段 | MultiTrainer 多阶段 |
| :--- | :--- | :--- |
| **组件构建时机** | 构造阶段一次性调用 `build_components` 返回 `RuntimeComponents` | 延迟执行：按阶段依次调用各阶段构建函数 |
| **模型与优化器状态** | 单一的 `self.model`, `self.optimizer`, `self.loss_fn` | 随阶段切换，每个阶段独享专用的 `model`, `optimizer`, `loss_fn`, `train_data` |
| **阶段间通信** | 无需阶段间通信 | 通过 `on_stage_complete` 和实例属性传递特征、中间状态与残差数据 |
| **早停与检查点** | 全局统一的 `EarlyStoppingConfig` | 每个阶段可独立配置 `EarlyStoppingConfig`，监控不同指标 |
| **测试推断** | 默认直接复用 `forward_pass` | 建议覆盖 `test_forward_pass`，实现多阶段联合推断 |

---

## 核心抽象与数据结构

多阶段训练器依托于 `utils.training` 中的两个核心类：

### 1. StageConfig
声明一个阶段的名称以及其延迟构建函数：
```python
@dataclass
class StageConfig:
    name: str  # 阶段标识（如 "km", "am", "pretrain" 等）
    build: Callable[[], StageComponents]  # 零参数可调用对象，返回 StageComponents
```

### 2. StageComponents
封装单阶段训练所需的独立运行时组件：
```python
@dataclass
class StageComponents:
    model: torch.nn.Module  # 该阶段的模型
    optimizer: torch.optim.Optimizer  # 该阶段的优化器
    loss_fn: torch.nn.Module  # 该阶段的损失函数
    train_data: Any  # 该阶段的训练数据（DataLoader 或 Dataset）
    val_data: Any | None = None  # 验证数据（可选）
    test_data: Any | None = None  # 测试数据（可选）
    epochs: int = 100  # 该阶段运行轮数
    lr_scheduler: Any | None = None  # 学习率调度器（可选）
    early_stopping: EarlyStoppingConfig | None = None  # 早停配置对象（非构造实例）
    max_clip_grad_norm: float | None = None  # 梯度裁剪阈值
    checkpoint_monitor: str | None = None  # 最佳模型监控指标（如 "auc"）
    checkpoint_mode: str | None = None  # 监控模式（"max" 或 "min"）
```

---

## 完整实现步骤

### 第一步：定义多阶段模型配置

继承 `ModelConfig`，在字段中定义各个阶段特有的超参数：

```python
from dataclasses import field
from typing import Literal
from utils.config import ModelConfig
from utils.core import register_model_config


@register_model_config("ABKT")
class ABKTConfig(ModelConfig):
    """ABKT 多阶段超参数配置。"""

    # KM 阶段超参数
    km_hidden_dim: int = 5
    km_lr: float = 1e-3
    km_epochs: int = 100
    km_patience: int = 10

    # AM 阶段超参数
    am_embedding_dim: int = 50
    am_lr: float = 1e-4
    am_epochs: int = 500
    am_patience: int = 10
    am_lambda: float = 0.1  # 正则化权重

    # 多阶段融合方式
    combine_mode: Literal["add", "mul"] = "add"
    batch_size: int = 128
```

### 第二步：继承 MultiTrainer 并声明阶段列表

训练器在 `__init__` 中只做跨阶段通用的数据预备（如解析 Q 矩阵与用户交互记录），通过 `build_stages` 声明阶段顺序：

```python
from utils.core import register_trainer
from utils.training import MultiTrainer, StageComponents, StageConfig


@register_trainer("ABKT")
class ABKTTrainer(MultiTrainer):
    def __init__(self, rc, data_src, exp_manager=None):
        # 1. 准备跨阶段共享数据（在调用 super 之前）
        from .ABKT_data import ABKTModelData

        self.model_data = ABKTModelData(data_src)
        self.data = self.model_data.prepare_data(rc)

        # 2. 跨阶段状态槽位（将在阶段完成时由 hook 填充）
        self.km_model = None
        self.am_model = None
        self.am_train_triplets = None
        self.am_test_triplets = None

        super().__init__(rc, data_src, exp_manager)

    def build_stages(self) -> list[StageConfig]:
        """返回阶段执行顺序。构建器函数延迟执行。"""
        return [
            StageConfig(name="km", build=self._build_km),
            StageConfig(name="am", build=self._build_am),
        ]
```

### 第三步：实现各个阶段的延迟构建方法

每个阶段独立组装网络、优化器与数据集：

```python
    def _build_km(self) -> StageComponents:
        """构建第一阶段：KM 知识模块。"""
        m = self.run_config.model
        from .ABKT_model import K_CMF

        # 实例化 KM 模型
        self.km_model = K_CMF(
            k_hidden_size=m.km_hidden_dim,
            skill_num=self.data["num_skills"],
            user_num=self.data["num_users"],
            item_num=self.data["num_items"],
            q_matrix=self.data["Q_matrix"].to(self.device_),
        )

        train_loader = self.data["km_train_loader"]
        val_loader = self.data["km_val_loader"]

        optimizer = torch.optim.Adam(self.km_model.parameters(), lr=m.km_lr)

        return StageComponents(
            model=self.km_model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            train_data=train_loader,
            val_data=val_loader,
            epochs=m.km_epochs,
            early_stopping=EarlyStoppingConfig(
                monitor="auc", mode="max", patience=m.km_patience
            ),
        )

    def _build_am(self) -> StageComponents:
        """构建第二阶段：AM 能力残差模块。

        注意：此时 KM 阶段已完全结束，self.am_train_triplets 已由 hook 准备就绪。
        """
        m = self.run_config.model
        from .ABKT_model import GMF

        if self.am_train_triplets is None:
            raise RuntimeError("AM 阶段构建前必须先计算 Boosting 残差！")

        self.am_model = GMF(
            n_users=self.data["num_users"],
            n_items=self.data["num_items"],
            embedding_k=m.am_embedding_dim,
        )

        # 基于 KM 残差创建 AM 数据加载器
        train_loader = create_optimized_dataloader(
            self.am_train_triplets,
            batch_size=m.batch_size,
            shuffle=True,
            device=self.device_,
        )
        val_loader = create_optimized_dataloader(
            self.am_test_triplets,
            batch_size=m.batch_size,
            shuffle=False,
            device=self.device_,
        )

        optimizer = torch.optim.Adam(self.am_model.parameters(), lr=m.am_lr)
        loss_fn = AMLoss(am_lambda=m.am_lambda, combine_mode=m.combine_mode)

        return StageComponents(
            model=self.am_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_loader,
            val_data=val_loader,
            epochs=m.am_epochs,
            early_stopping=EarlyStoppingConfig(
                monitor="auc", mode="max", patience=m.am_patience
            ),
        )
```

### 第四步：管理阶段生命周期钩子 (`on_stage_complete`)

通过覆盖 `on_stage_complete(name, result)`，在上一阶段结束时截获中间输出，计算残差或生成下一阶段输入：

```python
    def on_stage_complete(self, name: str, result) -> None:
        """阶段结束回调。"""
        if name == "km":
            logger.info("KM 阶段完成，正在基于 KM 输出计算 AM 阶段的 Boosting 残差...")
            self._compute_boosting_residuals()

    @torch.inference_mode()
    def _compute_boosting_residuals(self) -> None:
        """利用已训练好的 KM 模型推断全量训练与测试序列，计算梯度与残差。"""
        self.km_model.eval()
        # 遍历训练数据，计算残差 g 与样本权重 w
        # 并将带有残差的样本构造成 Tensor 赋值给 self.am_train_triplets 和 self.am_test_triplets
        # ... 具体残差计算代码 ...
```

### 第五步：实现阶段化前向传播与损失计算

通过检查 `self._current_stage` 动态分支前向逻辑：

```python
    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """按当前运行阶段分发前向计算。"""
        if self._current_stage == "km":
            return self._forward_km(batch_data)
        elif self._current_stage == "am":
            return self._forward_am(batch_data)
        else:
            raise ValueError(f"未知的阶段: {self._current_stage}")

    def _forward_km(self, batch_data: tuple) -> dict:
        user_ids, items, corrects = batch_data
        # KM 阶段前向计算
        pred = self.km_model(user_ids, items)
        return {
            "y_hat": pred,
            "y_label": corrects.float(),
            "y_predict": (pred >= 0.5).int(),
            "y_score": pred,
            "y_prob": pred,
        }

    def _forward_am(self, batch_data: tuple) -> dict:
        # AM 阶段前向计算
        batch = batch_data[0] if isinstance(batch_data, tuple) else batch_data
        u_idx, i_idx, correct, km_pred = (
            batch[:, 0],
            batch[:, 1],
            batch[:, 2],
            batch[:, 3],
        )

        am_pred, u_norm, i_norm = self.am_model(u_idx.long(), i_idx.long())
        if self.run_config.model.combine_mode == "mul":
            final_pred = am_pred * km_pred
        else:
            final_pred = am_pred + km_pred

        return {
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
```

### 第六步：覆盖测试推断 (`test_forward_pass`)

所有训练阶段完成后，UniKT 的 `TestEvaluationCallback` 会调用 `test_forward_pass`。覆盖该方法以实现双模型联合推断：

```python
    def test_forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """多阶段联合测试推断。"""
        self.km_model.eval()
        self.am_model.eval()

        user_ids, items, corrects = batch_data
        user_ids = self._move_tensor_to_device(user_ids)
        items = self._move_tensor_to_device(items)
        corrects = self._move_tensor_to_device(corrects)

        # 1. 取得 KM 知识模块的预测
        km_pred = self.km_model(user_ids, items)

        # 2. 取得 AM 能力模块的残差预测
        am_pred, _, _ = self.am_model(user_ids, items)

        # 3. 联合推断
        if self.run_config.model.combine_mode == "mul":
            final_pred = am_pred * km_pred
        else:
            final_pred = am_pred + km_pred

        final_pred = final_pred.clamp(0.0, 1.0)
        return {
            "y_hat": final_pred,
            "y_label": corrects.float(),
            "y_predict": (final_pred >= 0.5).int(),
            "y_score": final_pred,
            "y_prob": final_pred,
        }
```

---

## 最佳实践与注意事项

:::{tip}
1. **早停指标与检查点解耦**：在 `StageComponents` 中，可设置 `checkpoint_monitor` 与 `early_stopping.monitor` 不同。例如在 ABKT 中，KM 阶段早停监控 `acc`，但检查点可指定保存验证集上 `auc` 最高时的模型权重：
   ```python
   StageComponents(
       ...
       early_stopping=EarlyStoppingConfig(monitor="acc", mode="max", patience=10),
       checkpoint_monitor="auc",
       checkpoint_mode="max",
   )
   ```
2. **显存主动释放**：当切换阶段时，若前序阶段的优化器或中间激活值体积较大，可在 `on_stage_complete` 中将其从 GPU 上移至 CPU，或调用 `torch.cuda.empty_cache()`，防止两阶段显存叠加导致 OOM。
:::
