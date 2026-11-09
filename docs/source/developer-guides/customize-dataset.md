# 自定义数据集

UniKT 采用标准化、强类型的数据管线，将来自不同平台的原始答题日志，统一转换成标准化 Parquet 序列与特征关系表。

通过继承 `DataSource` 并实现数据提取、转换与加载规范，开发者可以轻松将专有或公开数据集接入到 UniKT 生态中。基类通过标准模板方法驱动整个处理流水线：

1. **`load_src_data()`**：读取 CSV、TSV 或 JSON 等格式的原始交互文件。
2. **`clean_raw_data()`**：重命名列并过滤缺省值或非法交互。
3. **`transform_data()`**：重新映射紧凑 ID，提取技能列表，调用 `set_sequence_data()` 注册主表。
4. **划分与序列切分**：基类自动处理样本分层抽样、K 折标签注入，并分别生成题目级与技能级时序切片。
5. **持久化落盘**：将规范化序列写入 Parquet 并自动生成 `metadata.json`。

---

## 标准数据规范与列模式

UniKT 规范化后的交互数据通过 `self.set_sequence_data` 注册，必须包含以下核心列：

| 字段名称 | 数据类型 | 描述 | 约束 |
| :--- | :--- | :--- | :--- |
| `interaction_id` | `Int64` / `Int32` | 交互事件唯一主键 | 全局唯一，自增 |
| `user` | `Int32` | 重新编号后的学生 ID | 连续紧凑整数：`[0, num_users - 1]` |
| `question` | `Int32` | 重新编号后的题目 ID | 连续紧凑整数：`[0, num_questions - 1]` |
| `skills` | `List(Int32)` | 该道题目关联的知识点 ID 列表 | 连续紧凑整数：`[0, num_skills - 1]`，列表非空 |
| `label` | `Int32` | 作答结果标签 | 二值变量：`0` 错误或 `1` 正确 |

此外，可携带额外特征列，例如 `timestamp` 作答时间戳、`hint_count` 提示次数与 `attempt_count` 尝试次数等，切分时系统会按配置自动透传。

---

## 编写数据集处理器

在 `utils/data_process/` 目录下创建新的数据处理器文件（例如 `my_dataset.py`）。实现继承自 `DataSource` 的数据集类，并通过 `@register_data_source("<dataset_name>")` 进行装饰注册。

### 完整实现代码模板

```python
"""自定义数据集处理器模板。"""

import os
from typing import Any

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source
from utils.data_process.data_source import DataSource

logger = get_logger(__name__)


@register_data_source("my_dataset")
class MyDatasetData(DataSource):
    """自定义知识追踪数据集处理器。"""

    def __init__(self, args: Any) -> None:
        """初始化数据集元信息与文件路径。"""
        super().__init__(
            dataset="my_dataset",
            data_base_path=args.data.data_base_path,
            data_url="https://example.com/dataset/my_dataset.zip",  # 下载源 URL（可选）
            seed=args.general.seed,
        )
        self.args = args
        # 原始数据文件存放于 data/<dataset>/raw/
        self.raw_data_path = os.path.join(self.data_folder, "raw", "interactions.csv")

    @override
    def load_src_data(self) -> None:
        """从 raw 目录加载原始数据。推荐使用 Polars 提高加载性能。"""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"未找到原始数据文件: {self.raw_data_path}")

        logger.info(f"正在从 {self.raw_data_path} 加载原始数据...")
        self.raw_data = pl.read_csv(
            self.raw_data_path,
            ignore_errors=True,
            null_values=["NA", "null", ""],
        ).lazy()

    @override
    def clean_raw_data(self) -> None:
        """数据清洗与标准化字段重命名。"""
        if self.raw_data is None:
            self.load_src_data()

        logger.info("执行原始数据清洗与字段对齐...")
        data = self.raw_data.with_row_index("interaction_id")

        # 1. 重命名列为 UniKT 标准名称
        data = data.rename(
            {
                "student_id": "user",
                "problem_id": "question",
                "kc_id": "skill",
                "is_correct": "label",
                "submit_time": "timestamp",
            }
        )

        # 2. 过滤缺失值与异常作答（如非 0/1 的作答结果）
        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("question").is_not_null()
            & pl.col("skill").is_not_null()
            & pl.col("label").is_in([0, 1])
        )

        # 3. 按用户与时间戳全局排序
        data = data.sort(["user", "timestamp"])

        self.cleaned_raw_data = data.collect()
        logger.info(f"数据清洗完成，保留有效记录数: {len(self.cleaned_raw_data)}")

    @override
    def transform_data(self) -> None:
        """执行紧凑 ID 映射、知识点列表组装并交付标准序列。"""
        logger.info("执行数据特征转换与 ID 重映射...")
        if self.cleaned_raw_data is None:
            raise ValueError("必须先调用 clean_raw_data 后才能执行 transform_data")

        # 1. 建立题目 ID 映射（将原始 ID 映射到 [0, N-1]）
        question_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("mapped_question")
        )
        self._build_id_mapping(self.cleaned_raw_data, ["question"])
        mapped_data = self.cleaned_raw_data.join(
            question_df, on="question"
        ).with_columns(pl.col("mapped_question").cast(pl.Int32).alias("question"))

        # 2. 建立知识点映射并支持多知识点格式（例如 "1_5_9" 或单一 KC）
        # 提取全部唯一知识点并重编号
        unique_skills = mapped_data.select(pl.col("skill").cast(pl.Utf8)).unique()
        self._build_id_mapping(unique_skills, ["skill"])

        # 3. 映射学生 ID 到 [0, U-1]
        self._build_id_mapping(mapped_data, ["user"])
        mapped_data = self._apply_id_mapping(mapped_data, ["user"])

        # 4. 构建标准 List(Int32) 类型的 skills 列
        # 如果原始 skill 是字符串且包含分隔符，用 str.split 展开；否则直接转列表
        mapped_data = mapped_data.with_columns(
            self._map_interaction_skills(pl.col("skill").cast(pl.Utf8).str.split("_"))
        )

        # 5. 组装规范序列数据
        sequence_data = mapped_data.select(
            [
                "interaction_id",
                "skills",
                "user",
                "question",
                "label",
                "timestamp",
            ]
        )

        # 6. 保存题目-属性关系表（可选，供图神经网络或多任务模型使用）
        # self.relation_data = {"q_matrix": ...}

        # 7. 提交处理后的交互表
        self.set_sequence_data(sequence_data)
        logger.info("数据转换与紧凑映射完成。")
```

---

## 数据切分与保存机制

`DataSource` 内部封装了强大的切分流水线，子类**无需重复实现切分与持久化逻辑**，只需在 CLI 中驱动或直接调用基类内置方法：

1. **`add_kfold_labels(n_splits=5, test_ratio=0.2)`**：
   - 划分学生维度的训练折与留出测试集（按学生划分，杜绝跨折学生信息泄露）。
2. **`build_split_question_sequence_data()`**：
   - 将长交互切分成适合模型处理的固定最大长度序列，并记录题目级切分元信息。
3. **`build_split_skill_sequence_data()`**：
   - 当一道题目关联多个知识点时，展开知识点序列并对齐时间步切分。
4. **`save_data()` 与 `save_metadata()`**：
   - 自动将切分后的序列保存为高效的 Apache Parquet 格式，并将题目总数、知识点总数、序列最大长度、切分哈希值等元数据保存为 `metadata.json`。

---

## 命令行执行与验证

通过 UniKT 的通用数据处理入口 `data_process.py` 执行流水线：

### 1. 下载原始数据
```bash
python data_process.py download -d my_dataset
```

### 2. 执行 ETL 转换与 5 折序列生成
```bash
python data_process.py process -d my_dataset --data.kfold 5 --data.max_seq_len 200
```

如果模型需要评测窗口测试指标，可追加 `--extra windowlate`：
```bash
python data_process.py process -d my_dataset --extra windowlate
```

### 3. 查看输出目录
处理完成后，`data/my_dataset/` 目录结构如下：
```text
data/my_dataset/
├── raw/
│   └── interactions.csv
├── metadata.json                                      # 包含 num_users, num_questions, num_skills 等统计
├── my_dataset_sequence.parquet                        # 清洗后的学生完整时序交互表
├── my_dataset_split_question_sequence.parquet         # 题目级切分序列
├── my_dataset_split_skill_sequence.parquet            # 知识点级切分序列
└── my_dataset_windowlate.parquet                      # 窗口测试评估序列 (若指定 --extra windowlate)
```

### 4. 驱动模型训练
数据生成后，任何已注册的模型均可无缝使用该数据集：
```bash
python train.py -m GIKT -d my_dataset --data.fold 0
```
