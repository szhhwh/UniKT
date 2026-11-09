# train.py 命令行参考

`train.py` 是 UniKT 的核心训练入口，基于 `jsonargparse` 实现双遍反射解析，支持完整强类型分层参数配置。

## 基本语法

```bash
python train.py -m <ModelName> -d <DatasetName> [OPTIONS]
```

也可以通过 `--config` 引入 YAML 配置文件，并用命令行参数覆盖特定配置项：

```bash
python train.py --config configs/my_experiment.yaml --model.learning_rate 0.0005
```

---

## 核心参数与分层命名空间

### 1. 核心标识参数

| 参数 | 短标识 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `--experiment.model_name` | `-m` | `str` | **必需** | 已注册的模型名称（如 `DKT`, `AKT`, `GIKT`, `Mamba4KT`, `ABKT` 等） |
| `--data.dataset` | `-d` | `str` | **必需** | 已处理的数据集名称（如 `assistments09`, `ednet_kt1`, `algebra2005` 等） |
| `--config` | | `str` | `None` | 指定 YAML 配置文件路径，自动注入到 RunConfig 配置树中 |

---

### 2. `--general.*`：框架通用配置

控制日志、设备、随机种子、实验追踪与检查点保存等框架级行为：

| 参数 | 类型 | 默认值 | 描述与取值说明 |
| :--- | :--- | :--- | :--- |
| `--general.seed` | `int` | `42` | 全局随机种子，控制 PyTorch, NumPy, Python 随机源 |
| `--general.device` | `str \| null` | `None` | 计算设备，如 `cuda`, `cuda:0`, `cpu`。为 `None` 时自动检测 GPU |
| `--general.deterministic` | `bool` | `True` | 是否启用 PyTorch 确定性算法（确保完全复现，可能会微弱降低 GPU 吞吐） |
| `--general.log_dir` | `str \| null` | `None` | 运行日志与模型检查点输出目录；为 `None` 时自动保存至 `runs/normal/<timestamp>` |
| `--general.checkpoint_path`| `str \| null` | `None` | 断点续训权重路径（若指定，模型将从此检查点恢复权重与优化器状态） |
| `--general.cloud_tracking` | `bool` | `True` | 是否启用云端/本地实验追踪（SwanLab 或 W&B，由环境变量 `KT_TRACKING_BACKEND` 控制） |
| `--general.save_last_checkpoint` | `bool` | `True` | 是否在每个 Epoch 结束时保存最新权重检查点 `last_checkpoint.pth` |
| `--general.log_batch_metrics` | `bool` | `False` | 是否记录每批次的详细 loss 到 `batch_metrics_<phase>.csv` |
| `--general.skip_test` | `bool` | `False` | 训练完成后是否跳过最终测试集的推断评估 |
| `--general.cache` | `bool` | `False` | 是否启用模型数据预处理的磁盘缓存（缓存至 `.cache/`） |
| `--general.pin_memory` | `bool \| null`| `None` | DataLoader 的锁页内存开关。为 `None` 时根据 CUDA 自动启用 |
| `--general.progress` | `Literal` | `auto` | 终端进度条渲染模式：`auto`（TTY 时自动显示）、`rich`（强制显示）、`none`（关闭） |

---

### 3. `--data.*`：数据集与切分参数

控制数据集加载、K 折划分、序列截断预算与抽样：

| 参数 | 类型 | 默认值 | 描述与取值说明 |
| :--- | :--- | :--- | :--- |
| `--data.fold` | `int` | `0` | 当前训练与评估的折索引，取值范围 `[0, kfold - 1]` |
| `--data.kfold` | `int` | `5` | K 折交叉验证总折数（需 $\ge 2$） |
| `--data.test_ratio` | `float` | `0.2` | 独立留出测试集的学生比例 |
| `--data.data_base_path` | `str` | `./data` | 数据集根目录路径 |
| `--data.min_seq_len` | `int` | `3` | 序列最小有效交互长度，少于该长度的学生序列将被丢弃 |
| `--data.max_seq_len` | `int` | `200` | 序列切分的最大截断长度预算 |
| `--data.truncation_stage` | `Literal` | `post_expansion` | 知识点展开截断时机：`post_expansion`（展开后截断）或 `pre_expansion`（展开前截断） |
| `--data.sample_size` | `int \| null` | `None` | 绝对抽样学生数量（为 `None` 时使用全量数据） |
| `--data.sample_ratio` | `float \| null`| `None` | 抽样学生比例 `0.0 ~ 1.0`（优先级高于 `sample_size`） |
| `--data.sample_strategy` | `Literal` | `random` | 抽样策略：`random`（纯随机）、`stratified`（按作答数与正确率分层）、`time`（按时间） |

---

### 4. `--early_stopping.*`：早停机制配置

| 参数 | 类型 | 默认值 | 描述与取值说明 |
| :--- | :--- | :--- | :--- |
| `--early_stopping.monitor` | `Literal` | `auc` | 早停监控的验证集指标：`auc`, `acc`, `rmse`, `loss`, `auprc` |
| `--early_stopping.mode` | `str` | `max` | 优化目标方向：`max`（指标越大越好，如 auc/acc）或 `min`（指标越小越好，如 loss/rmse） |
| `--early_stopping.patience`| `int` | `10` | 容忍无改善的最大 Epoch 轮数，超过后触发早停退出 |
| `--early_stopping.min_delta`| `float` | `0.0` | 判定为有效提升的最小变化阈值 |

---

### 5. `--compile.*`：PyTorch 2.x 图编译加速

控制 `torch.compile` 对模型的 JIT 图编译优化：

| 参数 | 类型 | 默认值 | 描述与取值说明 |
| :--- | :--- | :--- | :--- |
| `--compile.compile` | `bool` | `False` | 是否启用 `torch.compile` 模型编译 |
| `--compile.compile_mode` | `Literal` | `default` | 编译优化模式：`default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `--compile.compile_fullgraph` | `bool` | `False` | 是否要求整个前向计算图完整捕获（无 Python 解释器回退） |
| `--compile.compile_dynamic` | `bool \| null`| `None` | 是否启用动态尺寸跟踪（为 `None` 时由 PyTorch 自动判定） |
| `--compile.compile_backend` | `str` | `inductor` | 编译后端引擎（默认 `inductor`） |

---

### 6. `--model.*`：模型超参数

`--model.*` 为多态命名空间，根据 `-m <ModelName>` 自动绑定对应模型的强类型配置类。所有模型均继承并包含以下基础训练参数：

| 参数 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- |
| `--model.epochs` | `int` | `150` | 最大训练轮数（基类默认 150，具体模型可覆盖） |
| `--model.batch_size` | `int` | `32` | 训练批大小（基类默认 32，具体模型可覆盖，如 GIKT 默认 128） |
| `--model.learning_rate`| `float` | `1e-3` | 初始学习率（默认 0.001） |
| `--model.weight_decay` | `float` | `0.0` | 优化器 L2 权重衰减系数 |

此外，各个模型特有的私有超参数也挂载在 `--model.*` 下。例如：
- 对于 **GIKT**：`--model.n_hop`, `--model.skill_neighbor_num`, `--model.embedding_dim`, `--model.hidden_neurons` 等。
- 对于 **AKT**：`--model.d_model`, `--model.n_blocks`, `--model.dropout`, `--model.kq_same` 等。
- 对于 **Mamba4KT**：`--model.d_state`, `--model.d_conv`, `--model.expand` 等。

可以通过 `python train.py -m <ModelName> -d <DatasetName> --help` 查阅特定模型的所有专有参数。

---

## 配置合并优先级

参数合并遵循以下严格优先级（自高向低覆盖）：

1. **命令行参数**（如 `--model.learning_rate 0.002`）
2. **`--config <path>` 指定的 YAML 文件**
3. **模型与框架的默认 Dataclass 默认值**

---

## 产出目录与无损复现

每次训练将在 `runs/normal/` 目录下生成一个带时间戳与实验签名的独立目录：

```text
runs/normal/GIKT_assistments09_20260520-232131_fold0_bs128/
├── run_config.yaml         # 本次运行的全量强类型配置归档（用于无损复现与评估）
├── run_metadata.yaml       # 运行元数据（包含模型参数量、优化器类型、GPU型号等）
├── run.log                 # 完整终端运行日志
├── best_model.pth          # 验证集最优检查点权重
├── last_checkpoint.pth     # 最后一轮完整状态检查点（含优化器，用于断点续训）
├── metrics_train.csv       # 训练阶段各轮次损失与指标记录
├── metrics_val.csv         # 验证阶段各轮次指标记录
├── metrics_test.csv        # 测试集最终评估指标
└── timing.csv              # 各阶段耗时分析
```

### 使用归档配置快速复现训练

任何一次训练生成的 `run_config.yaml` 均可通过 `--config` 直接加载复现：

```bash
python train.py --config runs/normal/GIKT_assistments09_20260520-232131_fold0_bs128/run_config.yaml
```
