# data_process.py 命令行参考

`data_process.py` 是 UniKT 数据管线的统一命令行工具，提供数据集下载（`download`）与清洗切分处理（`process`）两个标准子命令。

## 子命令总览

```bash
# 1. 下载原始数据集压缩包并解压至 data/<dataset>/raw/
python data_process.py download -d <dataset> [OPTIONS]

# 2. 执行 ETL 清洗、紧凑映射、K 折划分与标准 Parquet 切分
python data_process.py process -d <dataset> [OPTIONS]
```

---

## 1. `download` 子命令

从公开源或官方镜像自动下载原始数据集文件，并在下载完成后校验并解压到 `data/<dataset>/raw/` 目录。

### 参数说明

| 参数 | 短标识 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `--data.dataset` | `-d` | `str` | **必需** | 目标数据集标识符（如 `assistments09`, `ednet_kt1`, `junyi2015`） |
| `--data_url` | | `str \| null`| `None` | 自定义下载链接（若提供则覆盖数据集默认的内置下载链接） |
| `--force` | | `bool` | `False` | 是否强制重新下载（即使本地已存在原始文件） |
| `--max_retries` | | `int` | `3` | 下载失败时的最大重试次数 |
| `--num_threads` | | `int` | `4` | 多线程并发分块下载线程数 |
| `--data.data_base_path` | | `str` | `./data` | 数据集根目录路径 |

### 示例

```bash
# 使用内置镜像下载 ASSISTments 2009 数据集
python data_process.py download -d assistments09

# 强制重新下载并指定外部下载源
python data_process.py download -d assistments09 --force --data_url "https://my-mirror.org/assist09.zip"
```

---

## 2. `process` 子命令

对已下载的原始数据执行清洗、实体重编号、分层抽样、K 折交叉验证划分与长序列滑动切分。

### 核心参数说明

| 参数 | 短标识 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `--data.dataset` | `-d` | `str` | **必需** | 目标数据集标识符 |
| `--extra` | | `list[str]` | `[]` | 额外处理步骤。例如传入 `windowlate` 以构建窗口评测评估集 |
| `--data.kfold` | | `int` | `5` | K 折交叉验证划分总折数（设为 `1` 或 `0` 禁用 K 折划分） |
| `--data.test_ratio`| | `float` | `0.2` | 独立留出测试集的学生比例 |
| `--data.min_seq_len` | | `int` | `3` | 序列最小有效交互长度，交互数少于此值的学生将被过滤 |
| `--data.max_seq_len` | | `int` | `200` | 序列滑动切分的最大长度预算 |
| `--data.truncation_stage` | | `Literal` | `post_expansion` | KC 展开截断时机：`post_expansion`（展开后截断）或 `pre_expansion`（展开前截断） |
| `--data.data_base_path` | | `str` | `./data` | 数据集保存根路径 |
| `--general.seed` | | `int` | `42` | 划分 K 折与抽样时使用的全局随机种子 |

### 数据抽样参数（可选）

用于在超大规模数据集（如 EdNet-KT1）上快速生成小型子集以便调试开发：

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--data.sample_ratio` | `float \| null` | `None` | 按比例抽样学生用户（`0.0 ~ 1.0`） |
| `--data.sample_size` | `int \| null` | `None` | 按绝对数量抽样学生用户 |
| `--data.sample_strategy`| `Literal` | `random` | 抽样算法：`random`（纯随机）、`stratified`（按作答数与正确率分层）、`time`（按时间排序） |
| `--data.sample_attempts_bins` | `list[int]` | `[20, 100]` | 分层抽样时的作答次数区间划分阈值 |
| `--data.sample_correct_bins` | `list[float]` | `[0.4, 0.8]` | 分层抽样时的平均答对率区间划分阈值 |

### 示例

```bash
# 1. 标准全流程处理：生成 5 折切分与题目级/知识点级序列
python data_process.py process -d assistments09

# 2. 生成带有 windowlate 窗口测试评估集的数据
python data_process.py process -d assistments09 --extra windowlate

# 3. 在大型数据集上抽取 10% 学生并生成测试集
python data_process.py process -d ednet_kt1 --data.sample_ratio 0.1 --data.sample_strategy stratified
```

---

## 产出文件结构

处理完成后，标准数据集目录结构如下：

```text
data/assistments09/
├── raw/                                                    # 原始解压文件
│   └── skill_builder_data_corrected_collapsed.csv
├── metadata.json                                           # 数据集统计元数据与数据指纹
├── assistments09_sequence.parquet                          # 清洗后的学生完整时序交互表
├── assistments09_split_question_sequence.parquet           # 题目级标准切分序列
├── assistments09_split_skill_sequence.parquet              # 知识概念级标准切分序列
├── assistments09_relation_question_skill.parquet           # 题目-技能关联表
└── assistments09_windowlate.parquet                        # 窗口测试评估数据 (可选)
```

### `metadata.json` 核心字段示例

```json
{
  "dataset": "assistments09",
  "num_users": 3852,
  "num_questions": 17702,
  "num_skills": 123,
  "kfold_n_splits": 5,
  "test_ratio": 0.2,
  "max_question_seq_len": 200,
  "max_skill_seq_len": 200,
  "split_question_sequence_data_md5": "323c5bbe9d01547cab2dfbef5b4698f8",
  "split_skill_sequence_data_md5": "76d2d984a2ce84757491c868aaf203e6"
}
```
