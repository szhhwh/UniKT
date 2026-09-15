"""ASSISTments 2012 dataset handler."""

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


@register_data_source("assistments12")
class Assistments2012Data(DataSource):
    """ASSISTments 2012 dataset handler."""

    def __init__(self, args: Any) -> None:
        """Initialize the ASSISTments 2012 dataset handler."""
        super().__init__(
            dataset="assistments12",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments12.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "2012-2013-data-with-predictions-4-final.csv"
        )

    @override
    def load_src_data(self) -> None:
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = pl.read_csv(
            self.raw_data_path,
            encoding="latin1",
            ignore_errors=True,
            try_parse_dates=False,
            null_values=["NA", ""],
        ).lazy()

    @override
    def transform_data(self) -> None:
        """Clean data and build question_data and sequence_data."""
        logger.info("Processing ASSISTments 2012 data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping
        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )

        # Save question ID mapping
        question_map = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        self._id_mappings["question"] = question_map
        logger.info(f"Built question ID mapping: {len(question_map)} unique questions")

        # Apply question mapping
        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build normalized relation tables
        question_skill = mapped_data.select(["question", "skill"]).unique(
            subset=["question", "skill"], keep="first"
        )
        logger.debug(f"question_skill shape: {question_skill.shape}")

        question_assignment = mapped_data.select(["question", "assignment"]).unique(
            subset=["question", "assignment"]
        )
        logger.debug(f"question_assignment shape: {question_assignment.shape}")

        question_template = mapped_data.select(["question", "template"]).unique(
            subset=["question", "template"]
        )
        logger.debug(f"question_template shape: {question_template.shape}")

        # Build ID mappings from each relation
        self._build_id_mapping(question_skill, ["skill"])
        self._build_id_mapping(question_assignment, ["assignment"])
        self._build_id_mapping(question_template, ["template"])
        logger.info(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}, "
            f"templates={self._get_mapped_count('template')}"
        )

        # Apply ID mappings
        question_skill = self._apply_id_mapping(question_skill, columns=["skill"])
        question_assignment = self._apply_id_mapping(
            question_assignment, columns=["assignment"]
        )
        question_template = self._apply_id_mapping(
            question_template, columns=["template"]
        )

        # Build final sequence_data
        sequence_data = mapped_data.select(
            [
                "user",
                "question",
                "label",
                "attempt_count",
                "hint_count",
                "ms_first_response",
                "timestamp",
            ]
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
        if self.raw_data is None:
            self.load_src_data()

        # Drop skill text column
        data = self.raw_data.drop("skill")

        # Rename columns
        data = data.rename(
            {
                "user_id": "user",
                "problem_id": "question",
                "correct": "label",
                "assignment_id": "assignment",
                "skill_id": "skill",
                "template_id": "template",
                "start_time": "timestamp",
            }
        )

        data = data.unique()
        data = data.collect()

        # Parse timestamp as datetime and convert to Unix milliseconds
        data = data.with_columns(
            [
                pl.col("timestamp")
                .str.strptime(pl.Datetime, strict=False)
                .dt.epoch("ms")  # Convert to Unix milliseconds (int64)
                .alias("timestamp")
            ]
        )

        # Convert to global relative time (dataset-wise earliest timestamp as zero)
        data = data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        data = data.sort(["user", "timestamp"])
        data = data.with_columns([pl.col("user").cast(pl.Int32)])
        # Filter out rows with null skill early (these questions have no skill info)
        data = data.filter(pl.col("skill").is_not_null())
        data = data.filter(pl.col("label").is_in([0, 1]))

        # Exclude short sequences
        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["Assistments2012Data"]
