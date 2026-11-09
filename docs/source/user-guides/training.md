# 模型训练

本指南介绍如何在 UniKT 中启动知识追踪模型的训练流程，包括单折快速实验、训练过程控制（参数配置、早停策略与断点续训），以及运行产物的结构与日志输出。

---

## 单折快速训练

在算法研发与代码调试阶段，通常指定单一折（如第 0 折）进行快速试验。

### 启动训练命令

使用 `train.py` 启动训练，通过 `-m` 指定模型名称，`-d` 指定数据集：

```bash
python train.py -m GIKT -d assistments09 --data.fold 0
```

上述命令将以 `assistments09` 数据集的第 0 折作为测试集、第 4 折作为验证集启动 GIKT 模型的训练。UniKT 采用分层反射式 CLI 体系，所有参数结构与代码中的配置类树严格对齐。

### 核心参数配置

常用训练参数可以通过分层命名空间传入：

```bash
python train.py -m GIKT -d assistments09 \
  --data.fold 0 \
  --model.epochs 100 \
  --model.batch_size 128 \
  --model.learning_rate 0.001 \
  --model.weight_decay 1e-5 \
  --general.device cuda:0 \
  --general.seed 42 \
  --general.deterministic true
```

| 参数项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `-m, --experiment.model_name` | `str` | 必填 | 模型名称（如 `GIKT`, `DKT`, `AKT`, `Mamba4KT`） |
| `-d, --data.dataset` | `str` | 必填 | 数据集名称（如 `assistments09`, `ednet_kt1`） |
| `--data.fold` | `int` | `0` | 当前交叉验证折索引（`0` 到 `kfold - 1`） |
| `--model.epochs` | `int` | 模型定义 | 训练总轮数 |
| `--model.batch_size` | `int` | 模型定义 | 批次大小 |
| `--model.learning_rate` | `float` | 模型定义 | 优化器初始学习率 |
| `--model.weight_decay` | `float` | 模型定义 | 权重衰减系数（L2 正则化） |
| `--general.device` | `str` | 自动检测 | 计算设备，可指定 `cuda`, `cuda:0`, `cpu` 等 |
| `--general.seed` | `int` | `42` | 全局随机种子（同步设置 Python、NumPy、PyTorch 及 CUDA 种子） |
| `--general.deterministic` | `bool` | `true` | 是否启用 PyTorch 确定性算法（确保实验结果可精确复现） |

:::{tip} 查看特定模型的专属超参数
如需查看特定模型独有的超参数（例如 GIKT 的 `--model.agg_type` 或 `--model.gcn_layers`），可以在命令末尾加入 `-h` 查看该模型的反射参数帮助：
```bash
python train.py -m GIKT -h
```
:::

---

## 训练过程控制与早停机制

### 早停机制

UniKT 内置了基于验证集指标监控的早停机制，在指标不再提升时提前终止训练，避免算力浪费并自动保留最佳模型权重：

```bash
python train.py -m GIKT -d assistments09 \
  --data.fold 0 \
  --early_stopping.patience 10 \
  --early_stopping.monitor auc \
  --early_stopping.mode max \
  --early_stopping.min_delta 0.0005
```

| 早停参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--early_stopping.patience` | `int` | `10` | 容忍指标未提升的连续 Epoch 轮数。设为 `0` 时禁用早停 |
| `--early_stopping.monitor` | `str` | `auc` | 监控的目标指标，支持 `auc`, `acc`, `rmse`, `loss` |
| `--early_stopping.mode` | `str` | `max` | 优化方向：`max`（指标越大越好，如 AUC、ACC）或 `min`（指标越小越好，如 RMSE、Loss） |
| `--early_stopping.min_delta` | `float` | `0.0` | 判定为有效提升的最小变化绝对阈值 |

当早停触发时，训练循环将打印提示，且保存在磁盘上的 `best_model.pth` 始终对应监控指标表现最优的那一轮权重。

### 检查点恢复与断点续训

如果训练被意外中断，或者希望在保存的检查点基础上继续微调，可以使用 `--general.checkpoint_path` 参数：

```bash
# 1. 确保在初始训练时保存了最后一轮状态（默认已开启）
python train.py -m GIKT -d assistments09 \
  --general.save_last_checkpoint true

# 2. 从指定检查点恢复模型权重并继续训练
python train.py -m GIKT -d assistments09 \
  --general.checkpoint_path runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128/best_model.pth
```

---

## 产物结构与日志输出

### 终端训练日志输出

训练启动后，终端将输出进度日志与指标汇总：

```text
2026-11-09 12:00:00 INFO     __main__: Experiment directory: runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128
2026-11-09 12:00:00 INFO     utils.training.base_trainer: Model: GIKT  Dataset: assistments09  Fold: 0
2026-11-09 12:00:01 INFO     utils.training.base_trainer: Trainer built successfully (Parameters: 412,850, Device: cuda:0)
2026-11-09 12:00:01 INFO     utils.training.base_trainer: Epoch 1/150
Epoch 1/150 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:05 • Train Loss: 0.5421
2026-11-09 12:00:06 INFO     utils.training.base_trainer: Epoch 1/150 took 5.21s (train 4.89s, val 0.32s)
2026-11-09 12:00:06 INFO     utils.training.checkpoint: Best validation AUC improved (0.0000 -> 0.7241). Checkpoint saved to runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128/best_model.pth
...
2026-11-09 12:03:15 INFO     utils.training.callbacks: Early stopping triggered at epoch 35. Best epoch: 25 (AUC: 0.7682)
2026-11-09 12:03:16 INFO     utils.training.base_trainer: Test metrics: AUC=0.7654, ACC=0.7312, RMSE=0.4182
```

### 运行产物结构

每次训练会在 `runs/normal/<run_id>/` 目录下生成完整的实验归档文件：

```text
runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128/
├── best_model.pth          # 验证集监控指标最优时的模型权重
├── last_checkpoint.pth     # 最后一轮训练完成后的权重（若启用）
├── run_config.yaml         # 类型完整的配置归档（支持无损重跑与配置复现）
├── run_metadata.yaml       # 运行元数据（包含参数量、硬件环境、PyTorch 版本等）
├── metrics_train.csv       # 训练集各轮次 Loss 与评价指标记录
├── metrics_val.csv         # 验证集各轮次指标记录
├── metrics_test.csv        # 测试集最终评价指标记录
├── run.log                 # 完整控制台日志记录
└── evaluate/               # 若运行了 evaluate.py，生成的独立评估日志目录
```
