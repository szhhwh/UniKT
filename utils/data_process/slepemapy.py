"""Slepemapy (Geography) dataset handler."""

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


@register_data_source("slepemapy")
class SlepemapyData(DataSource):
    """Slepemapy (Geography) dataset handler.

    The Slepemapy dataset is a geography learning dataset where students
    answer questions about geographic locations.

    Raw data columns:
        - id: Row ID
        - user: Student ID
        - place_asked: The place being asked about (question)
        - place_answered: The student's answer
        - type: Question type (1 or 2)
        - inserted: Timestamp (datetime string)
        - response_time: Response time in milliseconds
        - place_map: Map related info
        - language: Language
        - options: Options presented
        - ip_country: IP country
        - ip_id: IP ID

    Mapping to standard format:
        - user → user
        - f"{place_asked}_{type}" → question
        - place_asked → skill
        - place_asked == place_answered → label (1=correct, 0=incorrect)
        - inserted → timestamp
        - type → assignment
    """

    def __init__(self, args: Any) -> None:
        """Initialize the Slepemapy dataset handler."""
        super().__init__(
            dataset="slepemapy",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/slepemapy.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(self.data_folder, "raw", "answer.csv")

    @override
    def load_src_data(self) -> None:
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = pl.scan_csv(
            self.raw_data_path,
            separator=";",
            null_values=[""],
        )

    @override
    def clean_raw_data(self) -> None:
        """Clean raw sequence data."""
        if self.raw_data is None:
            self.load_src_data()

        # Select only needed columns and rename
        data = self.raw_data.select(
            [
                pl.col("user"),
                pl.col("place_asked"),
                pl.col("place_answered"),
                pl.col("type"),
                pl.col("inserted").alias("timestamp"),
            ]
        )

        # Drop rows where place_answered is null
        data = data.filter(pl.col("place_answered").is_not_null())

        # Build label: correct if place_asked == place_answered
        data = data.with_columns(
            (pl.col("place_asked") == pl.col("place_answered"))
            .cast(pl.Int32)
            .alias("label")
        )

        # Build question from place_asked + type combination
        data = data.with_columns(
            (
                pl.col("place_asked").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("type").cast(pl.Utf8)
            )
            .alias("question")
            .cast(pl.Categorical)
        )

        # Drop place_answered, then rename remaining columns
        data = data.drop("place_answered")
        data = data.rename({"place_asked": "skill", "type": "assignment"})

        data = data.collect(engine="streaming")

        # Convert timestamp string to milliseconds
        data = data.with_columns(
            (
                pl.col("timestamp")
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                .cast(pl.Int64)
                / 1_000
            )
            .cast(pl.Int64)
            .alias("timestamp")
        )

        # Convert to global relative time (dataset-wise earliest timestamp as zero)
        data = data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        # Sort by user and timestamp
        data = data.sort(["user", "timestamp"])

        # Remove duplicates
        data = data.unique()

        # Ensure label is 0 or 1
        data = data.filter(pl.col("label").is_in([0, 1]))

        # Cast types
        data = data.with_columns(
            [
                pl.col("user").cast(pl.Int32),
                pl.col("skill").cast(pl.Int32),
                pl.col("assignment").cast(pl.Int32),
            ]
        )

        # Exclude sequences that are too short
        data = exclude_short_sequences(data, self.args.min_seq_len)

        logger.debug(
            f"Cleaned data: {data.shape[0]} rows, "
            f"{data['user'].n_unique()} users, "
            f"{data['question'].n_unique()} questions, "
            f"{data['skill'].n_unique()} skills"
        )

        self.cleaned_raw_data = data

    @override
    def transform_data(self) -> None:
        """Transform cleaned data into standard question_data and sequence_data."""
        logger.info("Processing Slepemapy data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping (question = "place_asked_type" string)
        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )

        question_map = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        self._id_mappings["question"] = question_map
        logger.debug(f"Built question ID mapping: {len(question_map)} unique questions")

        # Apply question mapping to sequence data
        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build normalized relation tables
        question_data = (
            self.cleaned_raw_data.select(["question", "skill", "assignment"])
            .unique(subset=["question"])
            .join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        question_skill = question_data.select(["question", "skill"])
        question_assignment = question_data.select(["question", "assignment"])

        # Build ID mappings from each relation
        self._build_id_mapping(question_skill, ["skill"])
        self._build_id_mapping(question_assignment, ["assignment"])
        logger.debug(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}"
        )

        # Apply ID mappings
        question_skill = self._apply_id_mapping(question_skill, columns=["skill"])
        question_assignment = self._apply_id_mapping(
            question_assignment, columns=["assignment"]
        )

        # Build sequence_data
        sequence_data = mapped_data.select(["user", "question", "label", "timestamp"])

        # Build ID mapping for user
        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])
        logger.debug(f"Built user ID mapping: {self._get_mapped_count('user')} users")

        # Store processed data
        self.relation_data = {
            "question_skill": question_skill,
            "question_assignment": question_assignment,
        }
        self.sequence_data = sequence_data


__all__ = ["SlepemapyData"]
