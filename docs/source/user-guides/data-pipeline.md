# 数据处理管线

UniKT 提供了一套高吞吐、强类型、抗数据泄露的数据处理管线，通过统一入口 `data_process.py` 完成原始数据的下载、格式转换、特征工程与 K 折划分。

---

## 工作流与执行命令

数据预处理分为两个主要阶段：**下载阶段 `download`** 与 **处理阶段 `process`**。完整的数据处理包含五个步骤：

1. **拉取原始数据**：通过 `data_process.py download` 从官方镜像或 CDN 下载原始压缩包，解压到 `data/<dataset>/raw/`。
2. **清洗与 ID 映射**：通过 `data_process.py process` 使用 Polars 过滤缺省值并为学生、题目与知识点重新分配紧凑自增编号。
3. **零泄露划分**：按学生总体答题量与正答率分层划分为 5 折，并独立保留 Held-out 测试集。
4. **序列截断与展开**：根据配置的序列长度限制进行滑窗切分或截断。
5. **规范化落盘**：统一导出标准化 Parquet 数据表与 `metadata.json` 数据集统计元信息。

### 1. 下载原始数据

```bash
# 下载指定数据集，自动解压至 data/<dataset>/raw/
python data_process.py download -d assistments09

# 强制重新下载，覆盖本地已有 raw 缓存
python data_process.py download -d assistments09 --force true

# 指定并发下载线程数
python data_process.py download -d assistments09 --num_threads 8
```

### 2. 执行预处理流水线

```bash
# 使用默认配置进行处理，采用 5 折交叉验证与默认序列长度范围
python data_process.py process -d assistments09

# 自定义序列长度与折数
python data_process.py process -d assistments09 \
    --data.min_seq_len 5 \
    --data.max_seq_len 150 \
    --data.kfold 5 \
    --data.test_ratio 0.2

# 启用 windowlate 滑窗特征增强
python data_process.py process -d assistments09 --extra windowlate
```

---

## 产物目录与文件结构

预处理完成后，数据集目录（以 `data/assistments09/` 为例）包含以下标准文件：

```
data/assistments09/
├── raw/                                                # 下载并解压的原始 CSV/TXT 文件
├── assistments09_sequence.parquet                      # 全局清洗后的学生交互完整时序表
├── assistments09_split_question_sequence.parquet       # 题目级切片序列表 (供 QuestionModelData 读取)
├── assistments09_split_skill_sequence.parquet          # 技能级切片序列表 (供 SkillModelData 读取)
├── assistments09_relation_question_skill.parquet       # 题目-技能关联二部图关系表
├── assistments09_relation_question_assignment.parquet  # (可选) 题目-作业集合关联表
├── assistments09_relation_question_template.parquet    # (可选) 题目-模板关联表
└── metadata.json                                       # 数据集元数据与文件哈希快照
```

---

## 核心输出文件详解

### 1. 全局交互表

记录学生按时间严格排序的完整作答历史，是生成切片序列的源头。

| 字段名称 | 类型 | 含义说明 |
| :--- | :--- | :--- |
| `interaction_id` | `Int64` | 全局唯一的交互事件 ID |
| `user` | `Int32` | 重新映射后的学生 ID，取值范围 `[0, num_users - 1]` |
| `question` | `Int32` | 重新映射后的题目 ID，取值范围 `[0, num_questions - 1]` |
| `skills` | `List(Int32)` | 该题目关联的技能 ID 列表（支持单题关联多技能） |
| `label` | `Int64` | 作答结果：`1` 表示正确，`0` 表示错误 |
| `timestamp` | `Int64` | 毫秒级相对时间戳（以该数据集最早事件时间为基准点 `0`） |
| `fold` | `Int32` | 数据划分归属：`-1` 代表留出测试集，`0` 到 `K-1` 代表交叉验证折 |
| `attempt_count` | `Int64` | 尝试作答次数（若原数据集无此特征则设为 `1`） |
| `hint_count` | `Int64` | 索取提示次数（若原数据集无此特征则设为 `0`） |
| `ms_first_response` | `Int64` | 首次作答耗时（毫秒） |

### 2. 切片序列数据

深度学习知识追踪模型通常无法直接处理极长的不规则序列（如数千步交互）。UniKT 会根据 `max_seq_len` 将每个长序列切分为固定预算的子序列：

- **`{dataset}_split_question_sequence.parquet`**：用于题目级模型（如 SimpleKTQ、AKT、DKT 等面向题目的实现）。一行对应一次题目交互，`skills` 保留为 `List(Int32)`。
- **`{dataset}_split_skill_sequence.parquet`**：用于技能级模型（如 DKT、DKVMN、GKT 等面向技能概念的模型）。若一道题关联多个技能，管线会在此表中展开为多行，并将 `skill` 投影为单值 `Int32`。

切片表特有字段：

| 字段名称 | 类型 | 含义说明 |
| :--- | :--- | :--- |
| `sequence_id` | `Int32` | 切片子序列的全局唯一标识符 |
| `seq_pos` | `Int64` | 该交互在当前子序列中的相对索引，从 `0` 到 `split_len - 1` |
| `fold` | `Int32` | 继承自对应学生的 `fold` 标签，确保切片后划分依然严谨对齐 |

### 3. 实体关系表

用于构建图神经网络拓扑结构（如 GIKT, SGKT, HDHKT）或 Q 矩阵。最核心的关系表为 `question_skill`：

| 字段名称 | 类型 | 含义说明 |
| :--- | :--- | :--- |
| `question` | `Int32` | 题目标识 |
| `skill` | `Int32` | 关联的技能概念标识 |

---

## 序列截断与展开策略

知识追踪领域的一大经典技术难点在于：**多技能题目在转换为技能序列时，截断应发生在 KC 展开前还是展开后？** UniKT 通过 `--data.truncation_stage` 提供了显式控制：

1. **`post_expansion`（默认，推荐）**：
   - 先对单次题目的多个技能进行行展开，然后再按 `max_seq_len` 进行切块截断。
   - **特点**：保证每个子序列中的技能行数绝对不超过 `max_seq_len`，张量对齐简单；但在极少数情况下，同一道题关联的多个技能可能会被边界切开在两个相邻子序列中。
2. **`pre_expansion`**：
   - 先对题目的交互事件进行截断切块（每块最多 `max_seq_len` 次答题），再在块内展开技能。
   - **特点**：保证同一道题的所有技能永远处于同一个子序列内；展开后的实际行数可能略大于 `max_seq_len`，框架运行时会自动根据 `metadata.json` 中记录的真实最大长度动态适配张量。

---

## 数据集采样策略

对于超大规模数据集（如 EdNet 或 Junyi），UniKT 提供了内置采样逻辑，在不破坏统计分布的前提下加速原型开发：

```bash
# 1. 绝对数量采样：随机抽取 1,000 名学生及其全部答题历史
python data_process.py process -d junyi2015 --data.sample_size 1000 --data.sample_strategy random

# 2. 比例采样：抽取 10% 的学生
python data_process.py process -d ednet_kt1 --data.sample_ratio 0.1

# 3. 分层采样：按作答量与答对率分箱保持分布一致性
python data_process.py process -d junyi2015 \
    --data.sample_size 5000 \
    --data.sample_strategy stratified \
    --data.sample_attempts_bins [20, 100, 500] \
    --data.sample_correct_bins [0.3, 0.7]
```

---

## 元数据快照

每次运行 `process` 后，框架自动在数据集根目录下写入 `metadata.json`。模型训练时将自动读取该文件，用于初始化 Embedding 层的词表大小及 DataLoader 的 Padding 上限：

```json
{
    "dataset": "assistments09",
    "num_users": 3852,
    "num_questions": 17702,
    "num_skills": 123,
    "min_seq_len": 3,
    "max_question_seq_len": 200,
    "max_skill_seq_len": 200,
    "truncation_stage": "post_expansion",
    "kfold_n_splits": 5,
    "test_ratio": 0.2,
    "random_seed": 42,
    "sequence_data_md5": "e307060db821a897bf188dfc821bd4a0",
    "relation_question_skill_md5": "c6d52e345bcf8edcb48f1b98ea785a4a"
}
```
