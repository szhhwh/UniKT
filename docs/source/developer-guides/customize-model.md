# 自定义模型

在 UniKT 中接入全新的知识追踪模型遵循“模块解耦、静态注册、强类型配置”的开发范式。每个模型以独立包的形式组织在 `model/<ModelName>/` 目录下，由三个标准代码文件构成。

本指南以构建基于循环神经网络的知识追踪模型 `MyKT` 为例，展示从网络搭建、数据适配到训练器装配的实现流程。

## 模块结构规范

在项目根目录下的 `model/` 路径中新建文件夹 `model/<ModelName>/`，包含三个标准文件：

```text
model/MyKT/
├── MyKT_model.py    # 纯净的 PyTorch 计算图，继承 nn.Module
├── MyKT_data.py     # 序列特征准备与批切分，继承 QuestionModelData 或 SkillModelData
└── MyKT_trainer.py  # 训练循环装配与超参数注册，继承 BaseTrainer 与 ModelConfig
```

三个核心文件的职责划分如下：

| 文件名 | 职责与基类 | 核心方法与属性 |
| :--- | :--- | :--- |
| `<Name>_model.py` | 纯净的 PyTorch 计算图，继承 `torch.nn.Module` | `forward(...)` 接收特征张量，输出预测 logits 或预测概率 |
| `<Name>_data.py` | 数据准备与批处理，继承 `QuestionModelData` 或 `SkillModelData` | `prepare_data(rc)`，使用 `self.load_sequences()` 加载打包序列并切分折 |
| `<Name>_trainer.py` | 模型配置与运行时组件装配，继承 `BaseTrainer` 与 `ModelConfig` | 注册 `@register_model_config` 与 `@register_trainer`，实现 `build_components` 与 `forward_pass` |

:::{note}
UniKT 采用 **AST 静态发现系统**。创建上述文件并标注对应装饰器后，系统在扫描代码时仅通过解析语法树完成按需索引，无需在 `model/__init__.py` 或其他位置手动引入新模块，即可直接在 `train.py` 与 `optuna_search.py` 等 CLI 中生效。
:::

---

## 第一步：编写网络结构 (`<Name>_model.py`)

网络文件是标准的 PyTorch `nn.Module`，专注于模型结构的前向计算，与训练基础设施解耦。

### 预测语义与对齐约定

知识追踪模型通常有两种经典的序列预测语义：
1. **Next-item 预测**：主流默认范式。在 $t$ 时刻根据历史答题特征预测 $t+1$ 时刻的作答正确率，模型输出张量长度通常为 $S-1$ 或 $S$。
2. **Same-position 预测**：适用于 DKT 或 DKVMN 等结构。模型在前向过程中利用全部 $S$ 个时刻的表征输出 $S$ 维预测；此时输入需保证因果掩码，第 $t$ 步的输出对应第 $t$ 步的目标标签。

在 `MyKT_model.py` 中，我们以标准的 Next-item 循环神经网络为例：

```python
"""MyKT: 自定义轻量循环知识追踪网络模型。"""

import torch
import torch.nn as nn


class MyKTNet(nn.Module):
    """MyKT 神经网络模型。

    Args:
        num_questions: 数据集中题目总数。
        num_skills: 数据集中知识点总数。
        embedding_dim: 嵌入向量维度。
        hidden_dim: LSTM 隐藏层神经元数。
        dropout: 随机失活概率。
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 题目与作答的嵌入层 (0 为 padding，真实题目从 1 开始)
        self.question_embed = nn.Embedding(
            num_questions + 1, embedding_dim, padding_idx=0
        )
        self.skill_embed = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=0)
        # 答题标签 (0 或 1) 嵌入
        self.response_embed = nn.Embedding(2, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        # 输入维度: 题目嵌入 + 作答嵌入
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        # 预测头：预测下一次答题对错概率的 logit
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        question: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        skill: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播计算。

        Args:
            question: 题目 ID 序列 [batch_size, seq_len]
            response: 作答结果序列 (0/1) [batch_size, seq_len]
            mask: 有效位置布尔掩码 [batch_size, seq_len]
            skill: 知识点 ID 序列（可选）[batch_size, seq_len]

        Returns:
            预测概率或 logits，形状为 [batch_size, seq_len] 或 [batch_size, seq_len - 1]
        """
        # 1. 编码交互特征：当前题目嵌入 + 当前作答嵌入
        q_emb = self.question_embed(question)  # [B, S, D]
        r_emb = self.response_embed(response)  # [B, S, D]
        interaction_emb = torch.cat([q_emb, r_emb], dim=-1)  # [B, S, 2D]
        interaction_emb = self.dropout(interaction_emb)

        # 2. 序列建模
        h, _ = self.lstm(interaction_emb)  # [B, S, H]

        # 3. 预测概率（经过 Sigmoid 激活输出 0~1 的预测值）
        logits = self.fc(h).squeeze(-1)  # [B, S]
        y_hat = torch.sigmoid(logits)

        return y_hat
```

---

## 第二步：编写数据适配器 (`<Name>_data.py`)

知识追踪数据加载器负责将数据集转换为 PyTorch `Dataset`，并通过 K 折交叉验证划分出训练集、验证集和测试集。

### 基类选择：QuestionModelData vs SkillModelData

UniKT 提供了两个高级数据基类：
- **`QuestionModelData`**：以“题目”为序列主键粒度（每道题目可能关联多个或一个 KC）。适合需要题目建模的模型（如 AKT, GIKT, SimpleKT）。
- **`SkillModelData`**：以“知识概念”为序列主键粒度（在多技能交互时会自动展开）。适合经典的技能级模型（如 DKT, DKVMN）。

### 数据加载与折切分契约

框架严格保证模型层**不直接操作原始文件路径或全局缓存**，而是统一调用基类内置方法：

1. `self.load_sequences(required=..., optional=...)`：加载当前数据集打包序列，返回 `PackedSequences` 对象。
   - `required`: 模型必需的交互列（缺失会立即抛出异常并提示缺少字段）。
   - `optional`: 可选特征列（存在时加载，不存在则优雅降级）。
2. `PackedSequences` 属性：
   - `question`: 题目 ID 二维数组 `[N, S]`。
   - `response`: 作答结果二维数组 `[N, S]`。
   - `mask`: 有效位置布尔掩码 `[N, S]`。
   - `skills`: 题目关联的知识点数组。
   - `partition(fold_idx)`：根据指定折索引自动生成划分索引，并可通过 `partition.apply(dataset)` 对样本集进行零拷贝切分。
3. `to_first_skill_ids(skills)`：针对一个题目包含多个技能的情况，提供快速提取主知识点 ID 的辅助函数。

在 `MyKT_data.py` 中编写数据集类与数据预处理逻辑：

```python
"""MyKT 模型数据加载与准备。"""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class MyKTDataset(Dataset):
    """MyKT 训练与验证样本集。"""

    def __init__(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        masks: torch.Tensor,
        skills: torch.Tensor | None = None,
    ):
        self.questions = questions
        self.responses = responses
        self.masks = masks
        self.skills = skills

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        q = torch.tensor(self.questions[idx], dtype=torch.long)
        r = torch.tensor(self.responses[idx], dtype=torch.long)
        m = torch.tensor(self.masks[idx], dtype=torch.bool)
        if self.skills is not None:
            s = torch.tensor(self.skills[idx], dtype=torch.long)
            return q, r, m, s
        return q, r, m


class MyKTModelData(QuestionModelData):
    """MyKT 数据准备处理器。"""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple[Dataset, Dataset, Any]:
        """按照 RunConfig 规范切分并组装数据集。

        Args:
            rc: RunConfig 强类型配置对象

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        fold_idx = rc.data.fold
        logger.info(f"Loading packed sequences for fold {fold_idx}...")

        # 1. 统一加载对齐后的序列数据（零拷贝）
        sequences = self.load_sequences()
        partition = sequences.partition(fold_idx)

        # 2. 提取核心张量
        questions = sequences.question
        responses = sequences.response
        masks = sequences.mask
        # 若需要提取主知识点：
        skills = to_first_skill_ids(sequences.skills)

        # 3. 包装为 Dataset 并基于折划分应用切分
        full_dataset = MyKTDataset(questions, responses, masks, skills)
        train_dataset, val_dataset, _ = partition.apply(full_dataset)

        # 4. 构建标准 Windowlate 窗口测试评估集（统一评估协议）
        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        logger.info(
            f"Data prepared: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test(window)={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset
```

---

## 第三步：实现模型配置与训练器 (`<Name>_trainer.py`)

在 UniKT 中，模型超参数与训练器装配通过强类型类进行约束。

### 1. 注册模型超参数 (`ModelConfig`)

模型参数必须继承 `utils.config.ModelConfig`，并通过 `@register_model_config("<ModelName>")` 注册。
- 基类 `ModelConfig` 已经内置了标准参数契约：`epochs`, `batch_size`, `learning_rate`, `weight_decay`。
- 子类只需声明特定超参数，并为其赋予默认值与类型注解。
- 若需支持自动化超参数搜索，可以在字段元数据中声明 `metadata={"optuna": {...}}`，Optuna 寻参时会自动据此推断搜索空间。

### 2. 注册训练器 (`BaseTrainer`)

训练器必须继承 `utils.training.BaseTrainer`，并通过 `@register_trainer("<ModelName>")` 注册。核心需要实现两个方法：

1. **`build_components(self, rc, data_src) -> RuntimeComponents`**：
   - 组装模型、优化器、损失函数、学习率衰减器和数据集。
   - 返回 `RuntimeComponents` 容器。
   - 所有运行时对象（Model、Optimizer、Loss）存入 `RuntimeComponents`；所有标量配置严格通过 `rc` 访问，避免在 `self` 上悬挂多余属性。
2. **`forward_pass(self, batch_data) -> dict`**：
   - 将批次数据送入 GPU 设备：`self._move_tensor_to_device(...)`。
   - 执行前向计算获得预测值。
   - 使用辅助方法 `self._extract_valid_predictions(...)` 将预测值与真实标签按掩码对齐（针对 Next-item 自动处理偏移与打平）。
   - 处理空批次安全防护：`self._handle_empty_batch(...)`。
   - 生成 0/1 二值化预测：`self._generate_binary_predictions(...)`。
   - 返回标准指标字典：
     ```python
     {
         "y_hat": y_hat,  # 预测概率 [N_valid]
         "y_label": y_label,  # 真实标签 [N_valid]
         "y_predict": y_predict,  # 离散二值预测 0/1 [N_valid]
         "y_score": y_hat,  # AUC 连续分数 [N_valid]
         "y_prob": y_hat,  # 校准概率 [N_valid]
     }
     ```

在 `MyKT_trainer.py` 中的完整实现如下：

```python
"""MyKT 训练器与模型配置。"""

from dataclasses import field

import torch
import torch.nn as nn

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["MyKTConfig", "MyKTTrainer"]


@register_model_config("MyKT")
class MyKTConfig(ModelConfig):
    """MyKT 超参数配置树。

    Args:
        embedding_dim: 题目与知识点的嵌入维度。
        hidden_dim: LSTM 隐状态特征维度。
        dropout: 失活概率。
        epochs: 训练轮数。
        batch_size: 批大小。
        learning_rate: 优化器初始学习率。
        weight_decay: L2 正则化系数。
    """

    embedding_dim: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    hidden_dim: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    epochs: int = 150
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    weight_decay: float = 1e-5


@register_trainer("MyKT")
class MyKTTrainer(BaseTrainer):
    """MyKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        """延迟装配模型、数据与优化器运行时实例。"""
        # 延迟局部导入，避免未使用的模型拖慢系统发现与 CLI 启动
        from model.MyKT.MyKT_data import MyKTModelData
        from model.MyKT.MyKT_model import MyKTNet

        # 1. 准备数据
        train_data, val_data, test_data = MyKTModelData(data_src).prepare_data(rc)

        # 2. 从数据集获取元信息
        metadata = data_src.get_metadata()
        m = rc.model

        # 3. 实例化模型并迁移到预定设备
        logger.info(f"Initializing MyKT model with dim={m.embedding_dim}...")
        model = MyKTNet(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            embedding_dim=m.embedding_dim,
            hidden_dim=m.hidden_dim,
            dropout=m.dropout,
        )

        # 4. 优化器与损失函数
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )
        loss_fn = nn.BCELoss()

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            max_clip_grad_norm=5.0,  # 梯度裁剪阈值
        )

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """执行单批次前向传播并严格对齐标签。

        Args:
            batch_data: MyKTDataset 返回的数据元组 (q, r, m, [s])

        Returns:
            标准化指标计算字典
        """
        if len(batch_data) == 4:
            question, response, mask, skill = batch_data
            skill = self._move_tensor_to_device(skill)
        else:
            question, response, mask = batch_data
            skill = None

        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # 1. 模型前向计算 [B, S]
        y_hat_full = self.model(question, response, mask, skill=skill)

        # 2. 预测截断对齐（Next-item: same_position=False 默认，对齐 t 时刻预测与 t+1 标签）
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=False
        )

        # 3. 空批次防护
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # 4. 生成 0/1 离散预测
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
```

---

## 第四步：测试与验证

开发完成后，无需任何繁复的手动注册步骤，直接运行以下命令验证集成状态：

### 1. 验证静态扫描注册
通过运行 K 折脚本的 `--list` 选项，检查模型是否被 UniversalRegistry 正确索引：

```bash
bash scripts/run_kfold.sh --list | grep MyKT
```

如果输出 `MyKT`，表示模型配置与训练器已自动注册成功。

### 2. 启动单折训练
运行单折训练，确保数据加载与前向计算无异常：

```bash
python train.py -m MyKT -d assistments09 --data.fold 0 --model.epochs 5 --model.batch_size 64
```

### 3. 运行超参数自动化搜索
利用定义的 `optuna` 元数据，一键测试超参数搜索流程：

```bash
python optuna_search.py -m MyKT -d assistments09 --optuna_search.metric auc
```

---

## 常见问题与避坑指南

:::{important}
1. **张量未放置到对应设备**：在 `build_components` 中构建图矩阵、先验知识矩阵时，不要使用 `self.device_`（此时模型尚未 `build()`）。应统一读取 `rc.general.device`，或使用 `self.model.register_buffer(...)` 交由 PyTorch 托管设备迁移。
2. **预测与标签对齐错位**：绝不要在模型内部手写 `y[:, :-1]` 与 `mask[:, 1:]` 的展平截断逻辑！必须统一调用 `self._extract_valid_predictions(y_hat_full, response, mask, same_position=...)`。如果是 DKT 这种在相同位置预测当前题目的模型，传入 `same_position=True`；如果是预测下一题的模型，传入 `same_position=False`。
3. **空批次崩溃**：在极端过滤或填充序列的末尾，偶尔会出现该 batch 内完全没有有效掩码点的情况。务必在 `_extract_valid_predictions` 后调用 `self._handle_empty_batch(y_hat, y_label)`，避免 BCELoss 遇到零维张量触发 CUDA 异常。
:::
