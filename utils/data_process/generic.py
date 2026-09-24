"""Generic user-uploaded CSV dataset (``generic_<slug>``).

标准五列 ``raw/interactions.csv``（``user_id, item_id, skill_id, correct``
[, ``timestamp``]）经与内置数据集相同的清洗/映射/K 折/序列化流程产出
标准 parquet，供 train.py 直接训练——这是 web-saas 自有数据流水线的
数据地基。可选 ``raw/skills.csv``（``skill_id, skill_name``）提供知识
点名称，供门户技能目录展示。

该数据集没有下载源：raw 目录由上传方准备，load_src_data 在文件缺失
时给出明确指引。
"""

from __future__ import annotations

import os
from typing import Any

import polars as pl

from utils.core import get_logger

from .data_source import DataSource

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["user_id", "item_id", "skill_id", "correct"]


class GenericCsvData(DataSource):
    """Generic five-column CSV dataset handler (``generic_<slug>``)."""

    def __init__(self, args: Any) -> None:
        """Initialize with the slug taken from ``args.dataset``."""
        super().__init__(
            dataset=args.dataset,
            data_base_path=args.data_base_path,
            data_url=None,
            seed=args.seed,
        )
        self.args = args
        self.interactions_path = os.path.join(self.raw_folder, "interactions.csv")

    @property
    def slug(self) -> str:
        """Dataset slug without the ``generic_`` prefix."""
        return self.dataset.removeprefix("generic_")

    def load_src_data(self) -> None:
        """Load the uploaded interactions CSV as a lazy frame."""
        if not os.path.exists(self.interactions_path):
            raise FileNotFoundError(
                f"未找到 {self.interactions_path}。generic 数据集没有下载源，"
                "请先放置标准五列 CSV（user_id, item_id, skill_id, correct"
                "[, timestamp]）到 raw/ 目录。"
            )
        logger.info(f"Loading generic interactions from: {self.interactions_path}")
        self.raw_data = pl.read_csv(
            self.interactions_path,
            infer_schema_length=0,
            null_values=[""],
        ).lazy()

    def clean_raw_data(self) -> None:
        """Validate and normalize the five standard columns."""
        if self.raw_data is None:
            self.load_src_data()
        df = self.raw_data.collect()

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"interactions.csv 缺少必需列: {missing}；"
                f"实际列: {list(df.columns)}"
            )

        df = df.drop_nulls(REQUIRED_COLUMNS)

        # correct 必须是 0/1（接受 True/False 的常见 1/0 化）
        correct = (
            pl.col("correct").cast(pl.String).str.strip_chars().str.to_lowercase()
        )
        bad = df.filter(
            ~correct.is_in(["0", "1", "true", "false"])
        ).height
        if bad > 0:
            raise ValueError(
                f"correct 列有 {bad} 行不是 0/1（示例: "
                f"{df.filter(~correct.is_in(['0','1','true','false']))['correct'].head(3).to_list()}）"
            )

        has_ts = "timestamp" in df.columns
        cleaned = df.select(
            pl.col("user_id").cast(pl.String).str.strip_chars().alias("user"),
            pl.col("item_id").cast(pl.String).str.strip_chars().alias("question"),
            pl.col("skill_id").cast(pl.String).str.strip_chars().alias("skill"),
            correct.replace({"true": "1", "false": "0"}).cast(pl.Int8).alias("label"),
            (
                pl.col("timestamp").cast(pl.Int64)
                if has_ts
                else pl.int_range(pl.len()).cast(pl.Int64)
            ).alias("timestamp"),
        )

        n_users = cleaned["user"].n_unique()
        n_skills = cleaned["skill"].n_unique()
        if n_users < 2:
            raise ValueError(f"至少需要 2 个学生，当前 {n_users}")
        if n_skills < 2:
            raise ValueError(f"至少需要 2 个知识点，当前 {n_skills}")

        logger.info(
            f"Generic[{self.slug}]: {cleaned.height} 交互, "
            f"{n_users} 学生, {cleaned['question'].n_unique()} 题目, {n_skills} 知识点"
        )
        self.cleaned_raw_data = cleaned

    def transform_data(self) -> None:
        """Build the question mapping, question_skill relation, and sequences."""
        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )
        self._id_mappings["question"] = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        mapped = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        question_skill = mapped.select(["question", "skill"]).unique(
            subset=["question", "skill"]
        )
        self._build_id_mapping(question_skill, ["skill"])
        question_skill = self._apply_id_mapping(question_skill, columns=["skill"])

        sequence_data = mapped.select(["user", "question", "label", "timestamp"])
        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])

        self.relation_data = {"question_skill": question_skill}
        self.sequence_data = sequence_data
