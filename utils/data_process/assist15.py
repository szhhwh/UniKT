"""ASSISTments 2015 skill builders dataset handler."""

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


@register_data_source("assistments15")
class Assistments2015Data(DataSource):
    """Assistments 2015 skill builders dataset handler.

    Dataset source: https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data

    Raw columns: user_id, log_id, sequence_id, correct

    Since the dataset lacks explicit skill and problem IDs, sequence_id is used
    as both question and skill (standard approach in KT literature).
    """

    def __init__(self, args: Any) -> None:
        """Initialize the ASSISTments 2015 dataset handler."""
        super().__init__(
            dataset="assistments15",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments15.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "2015_100_skill_builders_main_problems.csv"
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
        ).lazy()

    @override
    def transform_data(self) -> None:
        """Clean data and build question_data and sequence_data."""
        logger.info("Processing ASSISTments 2015 data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping
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

        # Apply question mapping
        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build normalized relation tables
        # assist15 has no real skill/assignment — use identity mapping
        question_skill = mapped_data.select(["question"]).unique(subset=["question"])
        question_skill = question_skill.with_columns(pl.col("question").alias("skill"))

        question_assignment = mapped_data.select(["question"]).unique(
            subset=["question"]
        )
        question_assignment = question_assignment.with_columns(
            pl.col("question").alias("assignment")
        )

        # Build sequence_data
        sequence_data = mapped_data.select(
            ["user", "question", "label", "attempt_count", "hint_count", "timestamp"]
        )

        # Build ID mappings for user
        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])

        self.relation_data = {
            "question_skill": question_skill,
            "question_assignment": question_assignment,
        }
        self.sequence_data = sequence_data

    def clean_raw_data(self) -> None:
        """Clean raw sequence data."""
        if self.raw_data is None:
            self.load_src_data()

        data = self.raw_data.rename(
            {
                "user_id": "user",
                "sequence_id": "question",
                "log_id": "timestamp",
                "correct": "label",
            }
        )

        # Filter out nulls in critical columns
        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("question").is_not_null()
            & pl.col("label").is_not_null()
        )

        # Binarize label
        data = data.with_columns(
            pl.when(pl.col("label") >= 1.0)
            .then(pl.lit(1, dtype=pl.Int32))
            .otherwise(pl.lit(0, dtype=pl.Int32))
            .alias("label")
        )

        # Add default columns for attempt_count and hint_count
        data = data.with_columns(
            pl.lit(0, dtype=pl.Int32).alias("attempt_count"),
            pl.lit(0, dtype=pl.Int32).alias("hint_count"),
        )

        data = data.unique().collect()

        # Convert to global relative time
        data = data.with_columns(
            pl.col("timestamp")
            .cast(pl.Int64)
            .sub(pl.col("timestamp").cast(pl.Int64).min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        data = data.sort(["user", "timestamp"])

        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["Assistments2015Data"]
