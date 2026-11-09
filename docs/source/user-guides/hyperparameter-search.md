# 超参数自动化搜索

本指南介绍如何使用 UniKT 内置的 Optuna 超参数搜索运行器（`optuna_search.py`），对知识追踪模型的超参数进行单目标贝叶斯调优或多目标帕累托前沿搜索，以及如何断点恢复试验并分析搜索结果数据库。

## 目标与适用场景

- 针对特定数据集寻找最佳学习率、批大小、嵌入维度、层数或模型专属超参数。
- 支持单目标优化（如最大化 AUC、最小化 RMSE）与多目标优化（如同时兼顾 AUC 与 RMSE）。
- 支持试验异常中断后的无损断点恢复。
- 搜索完成后自动生成最佳参数配置 `best_run_config.yaml`，便于直接一键复现重跑。

## 前置条件

1. **环境准备**：Pixi 环境已内置安装 Optuna。
2. **数据就绪**：目标数据集已完成预处理。
3. **搜索空间元数据已注册**：各模型的搜索空间直接在模型专属的 `ModelConfig` 类中通过 `metadata={'optuna': ...}` 声明。无需手动编写或维护冗余的 JSON 参数文件。

## 运行超参数搜索

### 1. 单目标优化

最常用的场景是最大化验证集 AUC：

```bash
python optuna_search.py -m GIKT -d assistments09 --optuna_search.metric auc
```

此命令将读取 `configs/optuna/optuna_config.yaml` 中的全局配置，并结合 GIKT 模型定义的搜索空间执行试验。

### 2. 多目标帕累托优化

当需要同时权衡 AUC 与 RMSE 时，可以传入以逗号分隔的多个指标：

```bash
python optuna_search.py -m GIKT -d assistments09 --optuna_search.metric auc,rmse
```

```{note}
当传入多个指标时，系统自动切换为多目标优化（例如 `auc` 设为 `maximize`，`rmse` 设为 `minimize`）。进行多目标搜索时，建议将采样器设置为 `nsgaii`。
```

### 3. 命令行参数对照

| 参数项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `-m, --experiment.model_name` | `str` | 必填 | 待调优的模型名称（如 `GIKT`, `AKT`） |
| `-d, --data.dataset` | `str` | 必填 | 数据集名称（如 `assistments09`） |
| `--optuna_search.optuna_config` | `str` | `./configs/optuna/optuna_config.yaml` | Optuna 全局配置文件路径 |
| `--optuna_search.metric` | `str` | `auc` | 目标指标。支持 `auc`, `acc`, `rmse`, `loss`, `auprc`。多个指标用逗号分隔 |
| `--optuna_search.resume` | `str` | `null` | 从已有的运行目录恢复搜索（复用其中的 `study.db`） |
| `--optuna_search.keep_trial_artifacts` | `bool` | `false` | 是否保留每轮 Trial 的 SwanLab 上报、检查点与测试集评估（默认关闭以节省算力与存储） |
| `--optuna_search.output_dir` | `str` | `null` | 指定固定输出目录（供 Web 后端或自动化脚本确定性定位 `study.db`） |

除上述选项外，所有通用的 `--data.*` 和 `--general.*` 参数均可直接透传（例如 `--data.fold 0` 或 `--general.device cuda:0`）。

## 配置全局搜索参数：`optuna_config.yaml`

全局搜索控制文件位于 `configs/optuna/optuna_config.yaml`，核心字段如下：

```yaml
# Optuna study 级别配置
sampler: tpe             # 采样算法: tpe (Tree-structured Parzen Estimator) | random | nsgaii
sampler_kwargs:
  seed: 42
  n_startup_trials: 10   # 前 10 次试验采用随机采样建立初始先验

pruner: median           # 剪枝策略: median | percentile | successive_halving | hyperband | null
pruner_kwargs:
  n_startup_trials: 5    # 前 5 次试验不进行剪枝
  n_warmup_steps: 10     # 每个 Trial 前 10 个 Epoch 不剪枝

n_trials: 100            # 总试验次数
n_jobs: 1                # 并行进程数（单卡建议保持为 1）
timeout: null            # 超时时间（秒，null 表示无限制）
study_name: hyperparameter_search
db_url: null             # 数据库 URL（默认为本地 SQLite study.db）
save_dir: ./optuna_results
verbose: 1
```

### 常见配置调优建议

- **算力有限**：将 `n_trials` 调小至 `30`~`50`，并开启 `pruner: median` 以便尽早淘汰劣质参数组合。
- **多目标优化**：将 `sampler` 改为 `nsgaii`（非支配排序遗传算法）。
- **无剪枝纯评估**：若训练轮数较短或模型损失波动剧烈，可将 `pruner` 设为 `null` 避免误剪。

## 试验中断与断点恢复

如果超参数搜索因节点维护或意外断电中断，只需指定上一次的实验运行目录即可无缝继续：

```bash
python optuna_search.py -m GIKT -d assistments09 \
  --optuna_search.resume runs/hyperparam_search/GIKT_assistments09_20261109-120000_n_trials100
```

恢复时，UniKT 会自动连接已有目录下的 `study.db`，读取历史完成的 Trial 状态，并基于已积累的贝叶斯先验继续执行剩余的试验轮次。

## 产出物与结果分析

搜索完成后，实验目录将生成以下核心文件：

```text
runs/hyperparam_search/GIKT_assistments09_20261109-120000_n_trials100/
├── study.db                      # SQLite 格式的 Optuna 完整试验数据库
├── best_run_config.yaml          # 最优超参数组合构成的完整 RunConfig 配置文件
├── trials_history_gikt.csv       # 包含所有试验参数、得分、耗时、状态的表格
└── run.log                       # 搜索全流程日志记录
```

### 1. 使用最优配置一键重跑

系统生成的 `best_run_config.yaml` 已经将调优后的最优参数覆盖到位，且恢复了正常的检查点保存与测试集评估行为。你可以直接将其作为配置文件重新启动标准训练：

```bash
python train.py --config runs/hyperparam_search/GIKT_assistments09_20261109-120000_n_trials100/best_run_config.yaml
```

### 2. 通过 Optuna Dashboard 交互式查看

通过 Web 界面直观查看超参数重要性分析图、平行坐标图及帕累托前沿曲线：

```bash
optuna-dashboard sqlite:///runs/hyperparam_search/GIKT_assistments09_20261109-120000_n_trials100/study.db
```

在浏览器打开终端提示的地址（通常为 `http://127.0.0.1:8080`）即可交互式探索。

### 3. 使用 Python 脚本读取结果

也可以编写简单的脚本直接读取 `study.db`：

```python
import optuna

study = optuna.load_study(
    study_name="hyperparameter_search",
    storage="sqlite:///runs/hyperparam_search/GIKT_assistments09_20261109-120000_n_trials100/study.db",
)

print(f"完成的试验总数: {len(study.trials)}")
print(f"最佳指标值: {study.best_value:.4f}")
print("最佳超参数组合:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
```
