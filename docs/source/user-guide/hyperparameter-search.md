# 超参数搜索

使用 Optuna 进行自动化超参数优化。

## 概述

```{mermaid}
flowchart LR
 A[加载搜索空间] --> B[采样参数]
 B --> C[训练]
 C --> D[评估]
 D --> E[记录]
 E --> B
```

## 快速上手

```bash
# 使用默认配置运行（从 optuna_config.json 读取，100 个 trial）
python optuna_search.py -m GIKT -d assistments09
```

## 参数说明

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``-m, --experiment.model_name`` | 必填 | 模型名称 |
| ``-d, --data.dataset`` | 必填 | 数据集名称 |
| ``--optuna_search.optuna_config`` | ``./configs/optuna/optuna_config.yaml`` | Optuna 配置路径 |
| ``--optuna_search.metric`` | ``auc`` | 目标指标，逗号分隔（auc/acc/auprc/rmse/loss；多个启用多目标搜索） |
| ``--optuna_search.resume`` | 无 | 从 run 目录恢复搜索（复用 study.db） |
| ``--optuna_search.keep_trial_artifacts`` | ``false`` | 保留每 trial 的 SwanLab 追踪、checkpoint 与测试评估 |
| ``--optuna_search.output_dir`` | 时间戳目录 | 固定输出目录（study.db/trial 子目录/CSV），供外部调用方定位 |

其余 RunConfig 反射 flag（``--model.*`` / ``--data.*`` / ``--general.*``）与 ``train.py`` 相同；``optuna_search.py --help`` 查看完整参考。搜索空间由模型 ``ModelConfig`` 字段上的 ``metadata={'optuna': ...}`` 注解派生，无需单独的参数空间文件。


## Optuna 配置

在 ``configs/optuna/optuna_config.json`` 中配置搜索设置：

```json
{
 "sampler": "tpe",
 "sampler_kwargs": {
 "seed": 42,
 "n_startup_trials": 10
 },
 "pruner": "median",
 "pruner_kwargs": {
 "n_startup_trials": 5,
 "n_warmup_steps": 10
 },
 "n_trials": 100,
 "n_jobs": 1,
 "timeout": null,
 "directions": ["maximize"],
 "study_name": "gikt_hyperparameter_search",
 "db_url": null,
 "save_dir": "./optuna_results",
 "verbose": 1
}
```

**配置选项说明：**

| 选项 | 描述 |
| --- | --- |
| ``n_trials`` | 运行的 trial 数量 |
| ``n_jobs`` | 并行任务数（1 = 串行） |
| ``timeout`` | 超时秒数（null = 无限制） |
| ``directions`` | 优化方向（``maximize`` 或 ``minimize``） |
| ``db_url`` | 持久化数据库 URL（null = 内存模式） |


## 搜索空间配置

搜索空间在 ``configs/optuna/param_space_<model>.json`` 中定义为**数组**：

```json
[
 {
 "name": "lr",
 "type": "float",
 "low": 0.0001,
 "high": 0.01,
 "log": true,
 "default": 0.001
 },
 {
 "name": "hidden_dim",
 "type": "int",
 "low": 64,
 "high": 256,
 "log": true,
 "default": 100
 },
 {
 "name": "batch_size",
 "type": "categorical",
 "choices": [32, 64, 128, 256],
 "default": 128
 }
]
```

**分布类型：**

| 类型 | 必填字段 | 可选字段 | 描述 |
| --- | --- | --- | --- |
| ``float`` | ``low``, ``high`` | ``log`` | 浮点参数 |
| ``int`` | ``low``, ``high`` | ``log`` | 整数参数 |
| ``categorical`` | ``choices`` | - | 类别选择 |


**注意：** 每个参数必须包含 ``name`` 和 ``default`` 字段。

## 输出

结果保存在 ``runs/hyperparam_search/<study_name>_<timestamp>/``：

```
runs/hyperparam_search/gikt_hyperparameter_search_20240403-120000/
├── best_params.json # 最佳参数
├── trials_history_gikt.csv # 所有 trial 历史
└── study.log # 搜索日志
```

## 并行搜索

多个 worker 共享同一数据库运行：

```bash
# 终端 1 - 使用 SQLite 存储
python optuna_search.py -m GIKT -d assistments09 \
 --optuna_search.optuna_config configs/optuna/optuna_config_db.json

# 终端 2 - 相同命令
python optuna_search.py -m GIKT -d assistments09 \
 --optuna_search.optuna_config configs/optuna/optuna_config_db.json
```

**注意：** 并行搜索需要在配置中设置 ``db_url``：

```json
{
 "db_url": "sqlite:///optuna.db"
}
```

## 可视化

```python
import optuna

study = optuna.load_study(
    study_name="gikt_hyperparameter_search", storage="sqlite:///optuna.db"
)

# 优化历史
optuna.visualization.plot_optimization_history(study)

# 参数重要性
optuna.visualization.plot_param_importances(study)

# 平行坐标图
optuna.visualization.plot_parallel_coordinate(study)
```

## 最佳实践

1. **从宽范围开始**：初始使用较大的搜索范围
2. **迭代优化**：根据重要性分析缩小范围
3. **并行化**：使用 SQLite 存储进行多 worker 搜索
4. **监控**：通过 SwanLab 查看中间结果
5. **早停**：启用 pruner 提前终止无希望的 trial
