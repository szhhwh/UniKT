# 快速上手

本教程将带你走完 UniKT 的完整流程：从下载公开数据集、数据预处理，到训练经典深度知识追踪模型 DKT 并进行独立测试集评估。

---

## 准备工作

确保已按照[环境安装与配置](installation.md)安装了 Pixi 并进入环境：

```bash
pixi shell
```

---

## 步骤 1：下载与处理数据集

UniKT 内置了自动化数据流。以最经典且轻量的 **ASSISTments 2009** 数据集为例：

```bash
# 1. 自动从镜像源下载原始数据集压缩包并解压至 data/assistments09/raw
python data_process.py download -d assistments09

# 2. 清洗、构建题目-技能映射，并生成 5 折交叉验证划分的 Parquet 文件
python data_process.py process -d assistments09
```

处理完成后，在 `data/assistments09/` 目录下将生成如下标准化产物：
- `assistments09_sequence.parquet`：清洗后的完整学生答题交互记录。
- `assistments09_split_skill_sequence.parquet`：按技能粒度展开并切片后的序列（标注了所属 fold）。
- `assistments09_relation_question_skill.parquet`：题目与技能的对应关系二部图表。
- `metadata.json`：数据集全局元数据（用户数、题目数、技能数及校验 Hash）。

---

## 步骤 2：训练基准模型

使用 `train.py` 启动训练。这里以经典模型 **DKT** 为例，在第 0 折上进行训练：

```bash
python train.py -m DKT -d assistments09 --data.fold 0
```

:::{tip}
UniKT 采用简写参数：`-m` 指定模型名称，`-d` 指定数据集名称。其他参数采用分层命名空间，例如 `--data.fold 0` 指定第 0 折作为验证集。
:::

### 典型终端输出

启动后，终端将输出初始化流程与 Epoch 训练动态：

```text
2026-11-09 13:30:00 INFO     __main__: Experiment directory: runs/normal/DKT_assistments09_20261109-133000_fold0_bs128
2026-11-09 13:30:00 INFO     __main__: Building dataset: assistments09...
2026-11-09 13:30:00 INFO     utils.data_process.data_source: Loading split_skill_sequence data: ./data/assistments09/assistments09_split_skill_sequence.parquet [sequence_id, seq_pos, label, user, question, interaction_id, skill, fold]
2026-11-09 13:30:00 INFO     __main__: Initializing trainer for model: DKT...
2026-11-09 13:30:01 INFO     utils.training.base_trainer: RunConfig archive saved to runs/normal/DKT_assistments09_20261109-133000_fold0_bs128/run_config.yaml
2026-11-09 13:30:01 INFO     utils.training.base_trainer: Trainer built successfully (Parameters: 184,524, Device: cuda:0)
2026-11-09 13:30:01 INFO     __main__: Starting training...
2026-11-09 13:30:02 INFO     utils.training.base_trainer: Epoch 1/150
Epoch 1/150 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:03 • Train Loss: 0.5421
2026-11-09 13:30:05 INFO     utils.training.base_trainer: Epoch 1/150 took 3.42s (train 3.12s, val 0.30s)
2026-11-09 13:30:05 INFO     utils.training.checkpoint: Best validation AUC improved (0.0000 -> 0.7245). Checkpoint saved to runs/normal/DKT_assistments09_20261109-133000_fold0_bs128/best_model.pth
2026-11-09 13:30:05 INFO     utils.training.checkpoint: Checkpoint saved to runs/normal/DKT_assistments09_20261109-133000_fold0_bs128/last_checkpoint.pth
...
2026-11-09 13:32:15 INFO     utils.training.callbacks: Early stopping triggered after 24 epochs without improvement.
2026-11-09 13:32:15 INFO     utils.training.checkpoint: Loading best checkpoint from best_model.pth for test evaluation...
2026-11-09 13:32:16 INFO     utils.training.base_trainer: Test Set Evaluation Results: AUC=0.7482, ACC=0.7153, RMSE=0.4321
```

在训练过程中：
- 框架自动跟踪早停机制（默认 `patience=10`，监控验证集 AUC）。
- 每次验证集指标刷新历史最佳时，自动更新 `best_model.pth`。
- 每个 Epoch 结束时保存 `last_checkpoint.pth`。
- 训练结束后，自动加载最佳检查点并在留出测试集上运行最终评估。

---

## 步骤 3：独立测试集评估

如果需要重新评测某个已经训练完毕的模型，或者在不同的硬件（例如无显卡的 CPU 机器）上复现测试结果，可以使用 `evaluate.py`：

```bash
# 指定实验运行目录进行独立评估
python evaluate.py --evaluate.run_dir runs/normal/DKT_assistments09_20261109-133000_fold0_bs128
```

:::{note}
`evaluate.py` 会直接读取运行目录中的 `run_config.yaml` 归档文件，自动恢复完整的超参数、模型架构与数据集配置，因此**无需重复指定 `-m` 或 `-d`**。
:::

若需要在 CPU 上执行评估：

```bash
python evaluate.py --evaluate.run_dir runs/normal/DKT_assistments09_20261109-133000_fold0_bs128 --general.device cpu
```

控制台将输出测试集精确指标：

```text
2026-11-09 13:35:10 INFO     evaluate.py: Loading model weights from .../best_model.pth...
2026-11-09 13:35:11 INFO     evaluate.py: Running evaluation on test set...
2026-11-09 13:35:12 INFO     utils.training.base_trainer: Evaluation results: AUC=0.7482, ACC=0.7153, RMSE=0.4321
```

---

## 步骤 4：查看实验产物与追踪

每次训练运行都会在 `runs/normal/<run_id>/` 目录下生成自包含的实验归档文件：

```
runs/normal/DKT_assistments09_20261109-133000_fold0_bs128/
├── run_config.yaml        # 实验配置完整快照（无损复现的核心依据）
├── run_metadata.yaml      # 运行时元数据（参数量、GPU 型号、显存峰值等）
├── best_model.pth         # 验证集最优权重
├── last_checkpoint.pth    # 最终轮次检查点（用于断点恢复）
├── metrics_train.csv      # 训练轮次损失与指标明细
├── metrics_val.csv        # 验证轮次指标记录
├── timing.csv             # 各阶段耗时分析
└── run.log                # 完整终端运行日志
```

若在 `.env` 中启用了 SwanLab（`KT_TRACKING_BACKEND=swanlab`），可直接在 SwanLab Web 控制台查看动态绘制的损失曲线、AUC/ACC 变化以及多实验横向对比看板。
