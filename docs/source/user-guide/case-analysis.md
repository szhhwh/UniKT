# 案例分析

对训练好的模型进行案例分析——查看单个学生的答题序列和知识状态变化。

## 概述

案例分析的目的是**可视化模型对具体学生的预测质量**：模型在哪里对了、哪里错了、知识状态如何演变。 它通过三步工作流完成：推理 → 筛选学生 → 生成热力图。

```{mermaid}
flowchart LR
 A[inference<br/>加载 checkpoint<br/>批量推理] --> B[select<br/>按策略筛选<br/>目标学生]
 B --> C[plot<br/>生成知识状态<br/>热力图]
```

## 快速上手

```bash
# 第一步：推理
python case_analysis.py inference \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128

# 第二步：筛选
python case_analysis.py select \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128 \
 --case.selector diverse --case.num_users 10

# 第三步：可视化
python case_analysis.py plot \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128 \
 --case.selected_users diverse
```

:::{important}
三步必须按顺序执行——``select`` 依赖 ``inference`` 产出的 ``predictions.parquet``，``plot`` 依赖 ``select`` 产出的 ``selected_users.json``。
:::

--------------

(inference--批量推理)=

## inference —— 批量推理

加载 ``best_model.pth``，用验证集跑前向推理，保存预测结果。

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--case.run_dir`` | 必填 | run 目录路径（包含 ``best_model.pth`` 和 ``run_config.yaml``） |
| ``--case.checkpoint`` | ``best_model.pth`` | run 目录内的 checkpoint 文件名 |
| ``--case.sink`` | ``dataframe`` | 推理结果 sink 插件名（``CASE_SINKS`` 注册表） |
| ``--general.device`` | 归档值 | 设备覆盖（如 ``cpu``/``cuda``） |
| ``--model.batch_size`` | 归档值 | 推理 batch size 覆盖 |
| ``--data.data_base_path`` | 归档值 | 数据集根目录覆盖 |

其余 RunConfig 反射 flag 均可覆盖归档值（``case_analysis.py inference --help`` 查看完整参考）。

执行流程：从 ``run_config.yaml`` 归档恢复 ``RunConfig``（CLI flag 覆盖其上）→ 加载模型专属的 ``CaseAnalyzer`` → 遍历验证集批量推理 → 输出 parquet 文件。

推理完成后在 ``run_dir/case_analysis/`` 下生成：

```
case_analysis/
├── predictions.parquet # 每行一条答题记录（user_id, question_id, skill, label, prediction, position）
└── user_summaries.parquet # 每用户汇总（accuracy, auc, error_rate, num_attempts, calibration_error）
```

:::{note}
``CaseAnalyzer`` 继承 ``BaseCaseAnalyzer``（``utils/case_analysis/base_analyzer.py``），只加载 checkpoint 不创建 optimizer/loss——专为推理设计。如果 run 目录下没有 ``run_config.yaml``，命令会在解析阶段直接报错。
:::

--------------

(select--筛选学生)=

## select —— 筛选学生

从推理结果中按策略筛选目标学生。

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--case.run_dir`` | 必填 | run 目录路径 |
| ``--case.selector`` | ``diverse`` | 筛选策略：``diverse`` / ``extreme`` / ``random``（``CASE_SELECTORS`` 注册表） |
| ``--case.num_users`` | selector 默认（``20``） | 最多筛选多少学生 |
| ``--case.min_seq_len`` | selector 默认（``20``） | 学生最少答题数 |
| ``--case.min_error`` / ``--case.max_error`` | selector 默认（``0.1`` / ``0.9``） | 错误率窗口边界 |
| ``--case.min_confidence`` / ``--case.max_confidence`` | selector 默认（``0.3`` / ``0.95``） | 平均置信度窗口边界 |

未传的参数（默认值列 ``None``）回落到 selector 插件自身签名的默认值，CLI 不再手抄一份；只填窗口一端时，另一端自动用插件默认补齐。


### 三种策略

**diverse（多样采样，默认）**：将学生按错误率分 5 个桶（very_low → very_high），从每个桶均匀采样，覆盖不同能力层次。

```bash
python case_analysis.py select \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128 \
 --case.selector diverse --case.num_users 15
```

**extreme（极端案例）**：选择错误率最高的学生，适合分析模型失败模式。

```bash
python case_analysis.py select \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128 \
 --case.selector extreme --case.num_users 5
```

**random（随机采样）**：从符合条件的候选中随机抽取，适合快速抽查。

```bash
python case_analysis.py select \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128 \
 --case.selector random --case.num_users 10
```

### 筛选逻辑

1. 计算所有学生的 ``num_attempts`` / ``error_rate`` / ``avg_confidence``
2. 过滤掉 ``num_attempts < min_seq_len``、``error_rate`` 不在 ``[min_error, max_error]`` 或 ``avg_confidence`` 不在 ``[min_confidence, max_confidence]`` 的学生
3. 符合条件不足 ``num_users`` 则全部返回，否则按策略采样

结果保存到 ``run_dir/case_analysis/<selector>/selected_users.json``。

--------------

(plot--热力图可视化)=

## plot —— 热力图可视化

为选中的学生生成知识状态热力图。

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--case.run_dir`` | 必填 | run 目录路径 |
| ``--case.selected_users`` | 必填 | 策略名（``diverse``/``extreme``/``random``）或 JSON 文件路径 |
| ``--case.visualizer`` | ``heatmap`` | 可视化插件名（``CASE_VISUALIZERS`` 注册表） |
| ``--case.max_seq_len`` | ``None`` | 截断序列长度（仅影响绘图，与训练用的 ``--data.max_seq_len`` 无关） |


```bash
# 用策略名
python case_analysis.py plot \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128 \
 --case.selected_users diverse

# 用自定义 JSON 文件
python case_analysis.py plot \
 --case.run_dir runs/normal/DKT_assistments09_20260627-122915_fold0_bs128 \
 --case.selected_users /path/to/selected_users.json
```

### 热力图解读

每张热力图从上到下四个区域：

| 行 | 标签 | 内容 |
| --- | --- | --- |
| Question | ``q0``, ``q1``, ... | 每一步的题目 ID |
| Skill | ``c0``, ``c1``, ... | 每一步的技能 ID（多技能纵向排列，最多 3 个） |
| Resp | ✓ / ✗ | 实际答题正误（绿 ✓ / 红 ✗） |
| 热力图主体 | 色块 + 数值 | 每个技能的知识状态值（绿=高，红=低） |


图片保存为 ``run_dir/case_analysis/<selector>/figures/user_<user_id>_heatmap.png``，300 DPI。

:::{warning}
热力图需要 ``knowledge_states`` 列。大多数 DKT 类模型会将隐藏状态作为知识状态输出。如果该列全为 ``None``，``plot`` 会报错。
:::

--------------

## 自定义案例选择

可以直接用 ``utils.case_analysis`` 的公开函数在 Python 中自定义筛选：

```python
from utils.case_analysis import compute_user_metrics, get_user_sequence, load_case_results

df = load_case_results(
    "runs/normal/DKT_assistments09_20260627-122915_fold0_bs128/case_analysis/predictions.parquet"
)
metrics = compute_user_metrics(df)

# 选准确率最低的 5 个学生
worst = metrics.nsmallest(5, "accuracy")["user_id"].tolist()

# 获取某个学生的答题序列
seq = get_user_sequence(df, worst[0])
print(seq[["position", "question_id", "skill", "label", "prediction"]].head(10))
```

--------------

## 完整输出结构

```
runs/normal/DKT_assistments09_20260627-122915_fold0_bs128/
├── best_model.pth
├── run_config.yaml
├── metrics_train.csv / metrics_val.csv / metrics_test.csv / early_stopping.csv
└── case_analysis/
 ├── predictions.parquet
 ├── user_summaries.parquet
 ├── diverse/
 │ ├── selected_users.json
 │ └── figures/
 │ ├── user_42_heatmap.png
 │ └── ...
 ├── extreme/ ...
 └── random/ ...
```

## 模型支持

模型需要实现 ``@register_analyzer("NAME")`` 并继承 ``BaseCaseAnalyzer``（``utils/case_analysis/base_analyzer.py`` ），实现一个方法：

```python
def extract_case_data(self, batch_data, outputs) -> dict:
    return {
        "user_ids": ...,
        "question_ids": ...,
        "skills": ...,
        "labels": ...,
        "predictions": ...,
        "knowledge_states": ...,  # 可选，但热力图必需
        "mask": ...,
    }
```

:::{tip}
已有 34 个模型中的大部分实现了该方法。新模型参考 ``model/DKT/DKT_analyzer.py`` 创建即可。
:::

## 最佳实践

1. **先看 extreme**：找到模型表现最差的学生，优先修复失败模式
2. **再看 diverse**：覆盖不同能力层次，确认修复没有引入退化
3. **对比不同模型**：同一数据集、同一学生，对比两个模型的热力图
4. **截断长序列**：答题数超过 100 时用 ``--case.max_seq_len 100``，避免热力图过宽
