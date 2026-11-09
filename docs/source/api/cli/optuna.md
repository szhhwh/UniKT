# optuna_search.py 命令行参考

`optuna_search.py` 基于 [Optuna](https://optuna.org/) 提供了高度自动化、即插即用的超参数寻优系统。UniKT 从模型的 `ModelConfig` 中自动推断搜索空间，支持单目标优化、多目标联合优化与断点恢复。

## 基本语法

```bash
python optuna_search.py -m <ModelName> -d <DatasetName> [OPTIONS]
```

---

## 专属参数：`--optuna_search.*`

调优器特有的控制参数全部收敛在 `--optuna_search.*` 命名空间下：

| 参数 | 类型 | 默认值 | 说明与取值 |
| :--- | :--- | :--- | :--- |
| `--optuna_search.optuna_config` | `str` | `./configs/optuna/optuna_config.yaml` | Optuna 框架配置文件路径（指定试验总数 `n_trials`、采样器 `sampler`、剪枝器 `pruner` 等） |
| `--optuna_search.metric` | `str` | `auc` | 优化目标指标。可选值包括 `auc`, `acc`, `rmse`, `loss`, `auprc`。支持逗号分隔以启用**多目标联合寻优**（如 `auc,rmse`） |
| `--optuna_search.resume` | `str \| null` | `None` | 断点续寻目录。指定先前运行的目录（例如 `runs/hyperparam_search/...`），将复用其中的 `study.db` 继续采样 |
| `--optuna_search.keep_trial_artifacts` | `bool` | `False` | 是否保留每个 trial 的 SwanLab 追踪、权重检查点与测试集推断（默认关闭以节省大量显存与磁盘 I/O） |
| `--optuna_search.output_dir` | `str \| null` | `None` | 固定产物输出路径（用于 Web 平台或外部调度系统确定性定位 `study.db`） |

---

## 搜索空间自动推断机制

UniKT 无需手写冗长的 Optuna 建议代码。在编写模型的 `ModelConfig` 时，通过字段元数据声明 `metadata={"optuna": {...}}`，系统便会自动解析为 Optuna 搜索空间：

```python
@register_model_config("GIKT")
class GIKTConfig(ModelConfig):
    # 整数范围采样
    n_hop: int = field(
        default=3,
        metadata={"optuna": {"type": "int", "low": 1, "high": 5}},
    )
    # 对数尺度浮点采样
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    # 离散类别采样
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
```

支持的 `type` 包括：
- `int`：支持 `low`, `high`, `step`, `log`
- `float`：支持 `low`, `high`, `step`, `log`
- `categorical`：支持 `choices` 列表

---

## 常用命令示例

### 1. 基础单目标搜索
```bash
python optuna_search.py -m GIKT -d assistments09 --data.fold 0
```

### 2. 多目标联合优化
同时最大化 `auc` 并最小化 `rmse`（多目标时建议在 `optuna_config.yaml` 中配置 `sampler: nsgaii`）：
```bash
python optuna_search.py -m AKT -d assistments09 --optuna_search.metric auc,rmse
```

### 3. 从中断的搜索中恢复
```bash
python optuna_search.py -m GIKT -d assistments09 \
    --optuna_search.resume runs/hyperparam_search/GIKT_assist09_20260520_n_trials50
```

### 4. 覆盖单次 Trial 的训练轮数
可以在命令行中直接覆盖训练参数，加快搜索速度：
```bash
python optuna_search.py -m DKT -d assistments09 --model.epochs 30 --early_stopping.patience 5
```

---

## 产出物与最佳参数复现

搜索完成后，实验目录下将生成以下核心工件：

```text
runs/hyperparam_search/GIKT_assistments09_20261109-120000_n_trials100/
├── study.db                        # SQLite 数据库，保存所有 Trial 的参数与评估历史
├── best_run_config.yaml            # 自动生成的最佳超参数完整复现配置文件
├── best_params.yaml                # 最佳超参数扁平键值对
├── trials_history_gikt.csv         # 包含所有 Trial 的超参数与指标表格
├── optuna_config.yaml              # 本次搜索使用的超参数空间配置
├── param_importances.yaml          # 各超参数对目标指标的重要度评估
├── search_history.yaml             # 试验迭代历史轨迹
└── run.log                         # 搜索控制台日志
```

### 一键复现最佳超参数模型

搜索程序会自动去除搜索时的性能压制项（自动恢复 SwanLab 跟踪、检查点保存与测试集评估），生成标准干净的 `best_run_config.yaml`。只需一行命令即可使用最佳超参训练正式模型：

```bash
python train.py --config runs/hyperparam_search/GIKT_assistments09_20261109-120000_n_trials100/best_run_config.yaml
```
