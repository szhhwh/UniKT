# 命令行速查表

UniKT 基于 [jsonargparse](https://jsonargparse.readthedocs.io/) 构建了类型安全、分层解耦的命令行解析系统。所有 CLI 参数直接映射到强类型的 Dataclass（`RunConfig` 树），具备自动类型转换、合法性校验与层次化命名空间。

---

## 核心参数与简写

在所有主入口（`train.py`、`optuna_search.py` 等）中，为了提升输入效率，仅有两个高频参数保留单字母短标志：

| 短参数 | 完整层级参数 | 类型 | 说明 | 示例 |
| :--- | :--- | :--- | :--- | :--- |
| `-m` | `--experiment.model_name` | `str` | 注册的模型名称（严格大小写敏感） | `-m DKT` 或 `-m GIKT` |
| `-d` | `--data.dataset` | `str` | 注册的数据集名称 | `-d assistments09` 或 `-d ednet_kt1` |

---

## 分层参数速查表

UniKT 的参数按关注点划分到顶层命名空间中。使用 `--<节点>.<字段>` 语法传递参数：

### 1. 通用运行时参数 (`--general.*`)

控制设备分配、随机种子、实验工件与追踪平台：

| 参数 | 类型 | 默认值 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| `--general.device` | `str` | `null` (自动检测) | 指定执行硬件：`cuda`, `cuda:0`, `cpu` 等 |
| `--general.seed` | `int` | `42` | 全局随机种子（同步设置 Python、NumPy、PyTorch 及 CUDA） |
| `--general.deterministic` | `bool` | `true` | 是否启用确定性算法，确保结果可复现 |
| `--general.cloud_tracking` | `bool` | `true` | 是否启用云端追踪（SwanLab / W&B） |
| `--general.skip_test` | `bool` | `false` | 训练完成后是否跳过留出测试集的最终评估 |
| `--general.checkpoint_path`| `str` | `null` | 从指定检查点断点恢复训练（加载模型与优化器） |
| `--general.log_batch_metrics`| `bool` | `false` | 是否将每个 batch 的 loss 输出到 `batch_metrics_<phase>.csv` |

### 2. 数据与交叉验证参数 (`--data.*`)

控制序列切片、折数选择及截断策略：

| 参数 | 类型 | 默认值 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| `--data.fold` | `int` | `0` | 当前交叉验证选择的验证折（索引从 `0` 到 `kfold - 1`） |
| `--data.kfold` | `int` | `5` | 交叉验证的总折数（>= 2 生效） |
| `--data.test_ratio` | `float` | `0.2` | 留出测试集占全部用户的比例 |
| `--data.min_seq_len` | `int` | `3` | 过滤短于该交互长度的学生序列 |
| `--data.max_seq_len` | `int` | `200` | 序列切分的最大截断预算 |
| `--data.truncation_stage` | `choice` | `post_expansion` | KC 展开截断阶段：`pre_expansion` 或 `post_expansion` |
| `--data.sample_size` | `int` | `null` | 绝对采样用户数（用于轻量化快速冒烟测试） |
| `--data.sample_ratio` | `float` | `null` | 采样用户比例（0.0 ~ 1.0） |

### 3. 早停策略参数 (`--early_stopping.*`)

监控验证集指标并在性能停滞时提早终止训练：

| 参数 | 类型 | 默认值 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| `--early_stopping.monitor` | `choice` | `auc` | 监控目标指标：`auc`, `acc`, `rmse`, `loss`, `auprc` |
| `--early_stopping.mode` | `choice` | `max` | 优化方向：`max`（适合 auc/acc）或 `min`（适合 loss/rmse） |
| `--early_stopping.patience`| `int` | `10` | 容忍指标无改善的轮次数 |
| `--early_stopping.min_delta`| `float`| `0.0` | 判定为“有效改善”的最小绝对差值阈值 |

### 4. 模型基础与超参数 (`--model.*`)

基础参数（所有模型通用接口契约）：

| 参数 | 类型 | 默认值 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| `--model.epochs` | `int` | 各模型预设 (通常 100~200) | 训练轮次上限 |
| `--model.batch_size` | `int` | 各模型预设 (通常 64~256) | 批处理大小 |
| `--model.learning_rate` | `float` | 各模型预设 (通常 1e-3~1e-4) | 优化器初始学习率 |
| `--model.weight_decay` | `float` | `0.0` | 优化器 L2 正则化权重衰减系数 |

模型特定参数（根据 `-m` 动态绑定）：
- DKT / AKT / GIKT: `--model.hidden_dim 256`、`--model.dropout 0.2`
- Mamba4KT: `--model.d_state 16`、`--model.d_conv 4`
- GIKT: `--model.agg_type mean`、`--model.num_hops 2`

### 5. 图编译加速参数 (`--compile.*`)

封装 PyTorch 2.0+ `torch.compile` 能力：

| 参数 | 类型 | 默认值 | 作用与说明 |
| :--- | :--- | :--- | :--- |
| `--compile.compile` | `bool` | `false` | 是否开启 Inductor JIT 编译 |
| `--compile.compile_mode`| `choice` | `default` | 编译模式：`default`, `reduce-overhead`, `max-autotune` |

---

## 帮助文档与反射查看

### 1. 查看通用框架帮助

运行 `python train.py -h` 可查看框架层面的全局配置项：

```bash
python train.py -h
```

### 2. 查看特定模型的全部参数

当你指定了 `-m <模型名>` 后，jsonargparse 会自动解析该模型的专用 `ModelConfig` 类并将其参数动态挂载在 `--model.*` 命名空间下：

```bash
# 查看包含 GIKT 特定超参数的完整帮助文档
python train.py -m GIKT -h

# 查看包含 Mamba4KT 专用参数的完整帮助
python train.py -m Mamba4KT -h
```

### 3. 打印解析后的配置快照 (`--print_config`)

在实际启动前，若想确认所有默认值与覆盖项合并后的全量 YAML：

```bash
python train.py -m DKT -d assistments09 --data.fold 1 --model.batch_size 64 --print_config
```

---

## 配置文件支持 (`--config`)

UniKT 支持通过 YAML 配置文件进行批量化配置管理：

```bash
python train.py --config configs/my_experiment.yaml
```

CLI 参数的优先级高于配置文件。例如：在载入 YAML 的同时临时修改学习率和设备：

```bash
python train.py --config configs/my_experiment.yaml --model.learning_rate 0.0005 --general.device cuda:1
```

---

## 常用典型命令速查

```bash
# 1. 基础 GPU 训练（指定 Fold 0）
python train.py -m DKT -d assistments09 --data.fold 0

# 2. 指定显卡并调整批大小与学习率
python train.py -m AKT -d assistments09 --general.device cuda:1 --model.batch_size 64 --model.learning_rate 1e-4

# 3. 离线调试（无云端同步，关闭测试评估，采样 500 个学生）
python train.py -m DKT -d assistments09 --general.cloud_tracking false --general.skip_test true --data.sample_size 500

# 4. 修改早停策略（监控验证集 loss，容忍 15 个 epoch）
python train.py -m GIKT -d assistments09 --early_stopping.monitor loss --early_stopping.mode min --early_stopping.patience 15

# 5. 独立评估某个实验运行目录
python evaluate.py --evaluate.run_dir runs/normal/DKT_assistments09_20261109-133000_fold0_bs128

# 6. 使用 PyTorch Inductor 图编译加速训练
python train.py -m SimpleKT -d assistments09 --compile.compile true --compile.compile_mode default
```
