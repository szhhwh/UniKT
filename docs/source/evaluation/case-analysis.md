# 学生个体案例分析

本指南介绍如何在模型训练完成后，使用 UniKT 的案例分析工具（`case_analysis.py`）对单个学生的答题序列及知识状态演变进行细粒度微观诊断，通过标准的三步工作流生成知识掌握度热力图。

## 目标与适用场景

- 诊断模型在微观层面的预测行为：模型在哪些题目上给出了高置信度的错误预测，在哪些知识点上出现了震荡。
- 可视化学生在整条答题时间线上针对各个知识概念的掌握概率变化轨迹。
- 对比不同能力水平学生（高表现、低表现、多样性抽样）的认知状态差异。

## 前置条件

1. **环境准备**：已进入 Pixi 环境（`pixi shell`）。
2. **已完成模型训练**：存在至少一个已训练好的实验运行目录（包含 `run_config.yaml` 与检查点 `best_model.pth`）。例如：
   ```text
   runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128/
   ```

## 三步分析工作流

案例分析包含三个依序执行的步骤：

1. **`inference` 批量推断**：加载已训练权重并遍历验证集或测试集，导出全量答题预测与学生统计指标。
2. **`select` 策略抽样**：按照预设的多样性或极端值策略筛选代表性学生，输出候选学生列表 JSON。
3. **`plot` 状态可视化**：读取筛选出的学生交互序列，绘制知识掌握状态演变热力图并保存为 PNG 图像。

### 第一步：批量推理

加载已训练模型的检查点，遍历验证集或测试集数据进行前向推断，生成全量答题预测与学生级汇总指标：

```bash
python case_analysis.py inference \
  --case.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128
```

| 参数项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--case.run_dir` | `str` | 必填 | 训练运行目录，包含 `run_config.yaml` 与检查点权重 |
| `--case.checkpoint` | `str` | `best_model.pth` | 运行目录内待加载的检查点文件名 |
| `--case.sink` | `str` | `dataframe` | 数据持久化格式插件，默认使用标准化 DataFrame 存储 |

此步骤完成后，将在 `runs/normal/<run_id>/case_analysis/` 下生成：
- `predictions.parquet`：逐条交互的详细预测记录，包含 `user_id`、`question_id`、`skill`、`label`、`prediction` 与 `position` 等字段。
- `user_summaries.parquet`：按学生汇总的统计特征，涵盖答题次数、正确率、错误率与平均置信度。

### 第二步：筛选代表性学生

知识追踪数据集通常包含数千至数万名学生，逐一绘图既不现实也不利于定性分析。通过 `select` 子命令根据特定分析策略筛选代表性学生子集：

```bash
python case_analysis.py select \
  --case.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128 \
  --case.selector diverse \
  --case.num_users 10
```

#### 筛选策略 --case.selector

- **`diverse` 多样性采样（默认）**：
  将候选学生按错误率从低到高切分为极低、低、中、高、极高 5 个区间，从每个区间均匀抽取，确保抽样样本能覆盖不同认知水平的学生群体。
- **`extreme` 极端案例**：
  筛选错误率最高或置信度偏差最严重的极端学生，专门用于分析模型的失效模式。
- **`random` 随机抽样**：
  在满足序列长度与置信度过滤门槛的学生集合中进行等概率随机采样。

#### 过滤参数对照

| 过滤参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--case.selector` | `str` | `diverse` | 抽样算法：`diverse`, `extreme`, `random` |
| `--case.num_users` | `int` | `20` | 最多抽取的学生数量 |
| `--case.min_seq_len` | `int` | `20` | 学生交互序列最小长度，排除答题过少或信息不全的样本 |
| `--case.min_error` / `--case.max_error` | `float` | `0.1` / `0.9` | 错误率过滤区间 |
| `--case.min_confidence` / `--case.max_confidence` | `float` | `0.3` / `0.95` | 模型平均预测置信度区间 |

筛选完成后，结果保存在 `case_analysis/<selector>/selected_users.json` 中。

### 第三步：知识状态可视化

读取已筛选的学生列表，绘制知识概念随答题推进的状态演变热力图：

```bash
python case_analysis.py plot \
  --case.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128 \
  --case.selected_users diverse
```

| 参数项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--case.run_dir` | `str` | 必填 | 运行目录 |
| `--case.selected_users` | `str` | 必填 | 策略名称（如 `diverse`）或直接传入 `selected_users.json` 文件路径 |
| `--case.visualizer` | `str` | `heatmap` | 可视化插件名称，默认使用知识状态热力图 `heatmap` |
| `--case.max_seq_len` | `int` | `null` | 绘图序列长度截断，便于排版长序列 |

## 产出物与热力图解读

所有生成的图表均保存在对应策略的 `figures/` 目录下：

```text
runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128/case_analysis/
├── predictions.parquet
├── user_summaries.parquet
└── diverse/
    ├── selected_users.json
    └── figures/
        ├── user_128_heatmap.png
        ├── user_342_heatmap.png
        └── ...
```

```{figure} ../_static/case-study.webp
:alt: CIKT 在 Algebra 2005 数据集上的学生知识状态热力图。

CIKT 在 Algebra 2005 数据集上的学生知识状态轨迹。
```

### 热力图各视觉要素说明

在生成的 `user_<id>_heatmap.png` 中：

1. **水平横轴**：答题交互的时间步序列。每个步长标明了当前作答的题目或知识点。
2. **垂直纵轴**：数据集中相关的知识概念。
3. **颜色深浅**：代表模型预测该学生在当前时间步对特定知识点的掌握程度概率值，范围 0.0 至 1.0。颜色越深表示模型推断该学生对该概念掌握程度越高。
4. **作答标记**：
   - 绿色标记：学生实际作答正确，Label 为 1。
   - 红色标记：学生实际作答错误，Label 为 0。

```{tip}
通过观察连续做错某类题目时对应知识点颜色的变化，可以直观检验模型是否具备合理的“遗忘”或“负反馈修正”能力；当学生连续答对时，可以观察到知识状态的快速攀升。
```
