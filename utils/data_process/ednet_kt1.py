"""EdNet-KT1 dataset handler."""

import os
from typing import Any

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
    exclude_short_sequences,
)

logger = get_logger(__name__)


def _canonicalize_tags(tags: str, sep: str = ";") -> str | None:
    """Canonicalize tag string: deduplicate and sort.

    Args:
        tags: Tag string
        sep: Separator

    Returns:
        Canonicalized tag string, or None if input is empty
    """
    if tags is None:
        return None
    if not isinstance(tags, str):
        tags = str(tags)
    parts = [p.strip() for p in tags.split(sep) if p.strip()]
    if not parts:
        return None
    parts = sorted(dict.fromkeys(parts))
    return sep.join(parts)


@register_data_source("ednet_kt1")
class EdNetKT1Data(DataSource):
    """EdNet-KT1 dataset handler.

    Dataset source: https://github.com/riiid/ednet
    """

    def __init__(self, args: Any) -> None:
        """Initialize the EdNet-KT1 dataset handler."""
        super().__init__(
            dataset="ednet_kt1",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/EdNetKT1.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_folder = os.path.join(self.data_folder, "raw")

    @override
    def load_src_data(self) -> None:
        """Load raw data using Polars lazy evaluation with streaming."""
        self._validate_data_paths()

        question_answer_map = self._load_question_answers()
        response_pattern = os.path.join(
            self.raw_data_folder, "EdNet-KT1", "KT1", "*.csv"
        )

        logger.debug(f"Scanning CSV files from: {response_pattern}")

        lazy_df = (
            pl.scan_csv(
                response_pattern,
                schema_overrides={
                    "question_id": pl.Utf8,
                    "user_answer": pl.Utf8,
                    "timestamp": pl.Int64,
                },
                include_file_paths="__file_path__",
                glob=True,
            )
            .with_columns(
                pl.col("__file_path__")
                .str.extract(r"([^/]+)\.csv$", 1)
                .str.strip_prefix("u")
                .cast(pl.Int32)
                .alias("user_id")
            )
            .select(["user_id", "question_id", "user_answer", "timestamp"])
        )

        lazy_df = self._process_lazy_pipeline(lazy_df, question_answer_map)
        self.sequence_data_raw = lazy_df.collect(engine="streaming")
        logger.debug(f"Loaded {len(self.sequence_data_raw)} raw interactions.")

    def _validate_data_paths(self) -> None:
        """Validate that required data paths exist."""
        if not os.path.exists(self.raw_data_folder):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_folder}")
        logger.info(f"Loading raw data from: {self.raw_data_folder}")

        question_path = os.path.join(
            self.raw_data_folder, "EdNet-Contents", "questions.csv"
        )
        if not os.path.exists(question_path):
            raise FileNotFoundError(f"Cannot find: {question_path}")

        response_path = os.path.join(self.raw_data_folder, "EdNet-KT1", "KT1")
        if not os.path.exists(response_path):
            raise FileNotFoundError(f"Cannot find: {response_path}")

    def _load_question_answers(self) -> dict[str, str]:
        """Load question-answer mapping.

        EdNet uses letter answers (a, b, c, d), returns string mapping.

        Returns:
            Dictionary mapping question_id to correct_answer
        """
        question_path = os.path.join(
            self.raw_data_folder, "EdNet-Contents", "questions.csv"
        )
        lazy_questions = pl.scan_csv(question_path, try_parse_dates=False)

        qa_rows = (
            lazy_questions.select(["question_id", "correct_answer"])
            .filter(
                pl.col("question_id").is_not_null()
                & pl.col("correct_answer").is_not_null()
            )
            .with_columns(
                [
                    pl.col("question_id").cast(pl.Utf8),
                    pl.col("correct_answer").cast(pl.Utf8),
                ]
            )
            .sort("question_id")
            .unique(subset=["question_id"], keep="last")
            .collect()
            .to_dict(as_series=False)
        )

        question_answer_map = dict(
            zip(
                qa_rows["question_id"],
                qa_rows["correct_answer"],
            )
        )

        if not question_answer_map:
            raise ValueError("No valid question-answer mapping found for EdNet-KT1.")

        self.question_data_raw = lazy_questions.collect()
        return question_answer_map

    def _process_lazy_pipeline(
        self, lazy_df: pl.LazyFrame, question_answer_map: dict[str, str]
    ) -> pl.LazyFrame:
        """Process lazy DataFrame pipeline with vectorized operations.

        Args:
            lazy_df: Input LazyFrame
            question_answer_map: Question answer mapping

        Returns:
            Processed LazyFrame
        """
        lookup_lazy = pl.DataFrame(
            {
                "question_id": list(question_answer_map.keys()),
                "correct_answer": list(question_answer_map.values()),
            }
        ).lazy()

        return (
            lazy_df.join(lookup_lazy, on="question_id", how="inner")
            .with_columns(
                [
                    (pl.col("user_answer") == pl.col("correct_answer"))
                    .cast(pl.Int8)
                    .alias("label"),
                    pl.col("question_id").cast(pl.Utf8),
                ]
            )
            .select(["user_id", "question_id", "label", "timestamp"])
        )

    @override
    def transform_data(self) -> None:
        """Process and clean data."""
        logger.info("Processing EdNet KT1 data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping
        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )

        # Save question mapping for later use
        question_map = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        self._id_mappings["question"] = question_map
        logger.debug(f"Built question ID mapping: {len(question_map)} unique questions")

        # Apply question mapping
        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build question_data
        # Filter to only include questions that exist in cleaned_data
        valid_questions = set(self.cleaned_raw_data["question"].unique().to_list())

        question_meta_raw = (
            self.question_data_raw.rename(
                {
                    "question_id": "question",
                    "tags": "template_raw",
                    "bundle_id": "assignment",
                }
            )
            .filter(pl.col("question").is_in(valid_questions))
            .with_columns(
                [
                    pl.col("template_raw").cast(pl.Utf8).alias("template"),
                    pl.col("template_raw").cast(pl.Utf8).alias("skill"),
                ]
            )
            .select(["question", "skill", "assignment", "template"])
            .filter(pl.col("template").is_not_null())
        )
        logger.debug(f"Question metadata raw shape: {question_meta_raw.shape}")

        # Apply question mapping to question metadata
        question_meta = question_meta_raw.with_columns(
            pl.col("question").replace(question_map).cast(pl.Int32).alias("question")
        )
        logger.debug(f"Question metadata shape: {question_meta.shape}")

        # Build normalized relation tables
        # Relation 1: question_skill
        # EdNet skill format: "algebra;geometry" -> ["algebra", "geometry"]
        question_skill = (
            question_meta.with_columns(
                pl.col("skill").map_elements(_canonicalize_tags, return_dtype=pl.Utf8)
            )
            .with_columns(pl.col("skill").str.split(";").alias("skill_parts"))
            .explode("skill_parts")
            .with_columns(pl.col("skill_parts").cast(pl.String).alias("skill"))
            .select(["question", "skill"])
            .unique(subset=["question", "skill"], keep="first")
        )
        logger.debug(f"question_skill shape: {question_skill.shape}")

        # Relation 2: question_assignment
        question_assignment = question_meta.select(["question", "assignment"]).unique(
            subset=["question", "assignment"]
        )
        logger.debug(f"question_assignment shape: {question_assignment.shape}")

        # Relation 3: question_template
        question_template = question_meta.select(["question", "template"]).unique(
            subset=["question", "template"]
        )
        logger.debug(f"question_template shape: {question_template.shape}")

        # Build ID mappings from each relation independently
        self._build_id_mapping(question_skill, ["skill"])
        self._build_id_mapping(question_assignment, ["assignment"])
        self._build_id_mapping(question_template, ["template"])
        logger.debug(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}, "
            f"templates={self._get_mapped_count('template')}"
        )

        # Apply ID mappings independently
        question_skill = self._apply_id_mapping(question_skill, columns=["skill"])
        question_assignment = self._apply_id_mapping(
            question_assignment, columns=["assignment"]
        )
        question_template = self._apply_id_mapping(
            question_template, columns=["template"]
        )

        # Build final sequence_data
        # EdNet uses timestamp as order, no attempt_count/hint_count columns
        # Use placeholders for attempt_count/hint_count (set to 1 and 0 respectively) since they are required by our format
        sequence_data = (
            mapped_data.select(["user", "question", "label", "timestamp"])
            .with_columns(
                [
                    pl.lit(1).alias("attempt_count"),  # Placeholder
                    pl.lit(0).alias("hint_count"),  # Placeholder
                ]
            )
            .select(
                [
                    "user",
                    "question",
                    "label",
                    "attempt_count",
                    "hint_count",
                    "timestamp",
                ]
            )
        )

        # Build ID mapping for user in sequence_data
        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])
        logger.debug(f"Built user ID mapping: {self._get_mapped_count('user')} users")

        # Store processed data
        self.relation_data = {
            "question_skill": question_skill,
            "question_assignment": question_assignment,
            "question_template": question_template,
        }
        self.sequence_data = sequence_data

    def clean_raw_data(self) -> None:
        """Clean raw sequence data."""
        self.load_src_data()

        required_cols = ["user_id", "question_id", "label", "timestamp"]
        current_cols = self.sequence_data_raw.columns
        missing_cols = [col for col in required_cols if col not in current_cols]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in sequence data: {missing_cols}"
            )

        sequence_data = (
            self.sequence_data_raw.select(required_cols)
            .with_columns([pl.col("question_id").cast(pl.Utf8)])
            .rename({"question_id": "question", "user_id": "user"})
            .filter(
                pl.col("user").is_not_null()
                & pl.col("question").is_not_null()
                & pl.col("label").is_not_null()
            )
            .sort(["user", "timestamp"])
        )

        # Convert to global relative time (dataset-wise earliest timestamp as zero)
        sequence_data = sequence_data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        # Exclude sequences that are too short
        data = exclude_short_sequences(sequence_data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["EdNetKT1Data"]
