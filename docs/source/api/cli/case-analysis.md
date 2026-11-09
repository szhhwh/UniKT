# case_analysis.py 命令行参考

`case_analysis.py` 提供了面向已训练模型的细粒度学生个案推断、代表性样本筛选以及知识状态轨迹热力图渲染管线。

工作流分为三个标准步骤：

1. **`inference`**：全量前向推断，生成 `predictions.parquet` 与 `user_summaries.parquet`。
2. **`select`**：按策略筛选代表性学生，输出 `selected_users.json`。
3. **`plot`**：绘制单个学生的知识状态时序热力图，输出 PNG 图像。

---

## 步骤 1：运行模型推断 (`inference`)

加载已训练模型的检查点，执行推断并将各时刻的预测概率、真实标签与掌握度状态持久化到磁盘。

### 参数说明 (`--case.*`)

| 参数 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- |
| `--case.run_dir` | `str` | **必需** | 训练实验输出目录路径，包含 `run_config.yaml` 与检查点 |
| `--case.checkpoint` | `str` | `best_model.pth` | 加载的模型权重文件名 |
| `--case.sink` | `str` | `dataframe` | 数据落盘接收器插件名称，默认为 Parquet 格式 |

### 示例

```bash
python case_analysis.py inference \
    --case.run_dir runs/normal/HDHKT_assistments09_20260520_fold0
```

运行后将在实验目录下生成 `case_analysis/predictions.parquet`。

---

## 步骤 2：筛选代表性学生 (`select`)

基于答题准确率、平均置信度与序列长度等统计指标，从全量学生中自动挖掘具有代表性、对比性或极端表现的案例。

### 参数说明 (`--case.*`)

| 参数 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- |
| `--case.run_dir` | `str` | **必需** | 实验输出目录路径 |
| `--case.selector` | `str` | `diverse` | 筛选策略插件：`diverse` 多样性分层、`extreme` 两极化案例、`random` 随机抽样 |
| `--case.num_users` | `int` | `20` | 最多挑选的学生数量 |
| `--case.min_seq_len` | `int` | `20` | 学生交互序列最小答题长度要求 |
| `--case.min_error` | `float` | `0.1` | 学生作答错误率筛选下限 |
| `--case.max_error` | `float` | `0.9` | 学生作答错误率筛选上限 |
| `--case.min_confidence` | `float` | `0.3` | 模型平均预测置信度下限 |
| `--case.max_confidence` | `float` | `0.95` | 模型平均预测置信度上限 |

### 示例

```bash
# 采用多样性分层策略挑选 10 名典型学生
python case_analysis.py select \
    --case.run_dir runs/normal/HDHKT_assistments09_20260520_fold0 \
    --case.selector diverse \
    --case.num_users 10
```

筛选结果将保存为 `case_analysis/<selector>/selected_users.json`。

---

## 步骤 3：生成可视化热力图 (`plot`)

将筛选出的学生交互序列与知识状态演变渲染为高分辨率热力图。

### 参数说明 (`--case.*`)

| 参数 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- |
| `--case.run_dir` | `str` | **必需** | 实验输出目录路径 |
| `--case.selected_users` | `str` | **必需** | 筛选器名称（如 `diverse`）或直接指定 `selected_users.json` 文件路径 |
| `--case.visualizer` | `str` | `heatmap` | 可视化器插件名称，默认为知识掌握度热力图 `heatmap` |
| `--case.max_seq_len` | `int \| null` | `None` | 热力图横轴渲染的最大时间步截断，避免超长序列过密 |

### 示例

```bash
python case_analysis.py plot \
    --case.run_dir runs/normal/HDHKT_assistments09_20260520_fold0 \
    --case.selected_users diverse \
    --case.max_seq_len 100
```

热力图图片将输出到 `case_analysis/diverse/figures/user_<user_id>_heatmap.png`。

---

## 产出目录结构

三步分析完成后，案例分析的完整输出层次如下：

```text
runs/normal/HDHKT_assistments09_20261109-120000_fold0_bs128/
└── case_analysis/
    ├── predictions.parquet                 # 全量推断结果表
    ├── user_summaries.parquet              # 学生层级汇总指标表
    └── diverse/
        ├── selected_users.json             # 被选取的学生 ID 及其指标画像
        └── figures/                        # 各学生的热力图图像
            ├── user_42_heatmap.png
            ├── user_108_heatmap.png
            └── ...
```
