"""Junyi Academy 2015 dataset handler."""

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


@register_data_source("junyi2015")
class Junyi2015Data(DataSource):
    """Junyi Academy 2015 dataset handler.

    Dataset source: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198
    """

    def __init__(self, args: Any) -> None:
        """Initialize the Junyi 2015 dataset handler."""
        super().__init__(
            dataset="junyi2015",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/junyi2015.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_folder = os.path.join(self.data_folder, "raw")

    @override
    def load_src_data(self) -> None:
        """Load raw data using Polars lazy evaluation with streaming."""
        self._validate_data_paths()

        # Load exercise metadata table
        exercise_path = os.path.join(self.raw_data_folder, "junyi_Exercise_table.csv")
        logger.info(f"Loading exercise metadata from: {exercise_path}")
        self.exercise_data = pl.read_csv(exercise_path, try_parse_dates=False)

        # Load problem log using lazy scanning for streaming (large, ~25M rows)
        log_pattern = os.path.join(
            self.raw_data_folder, "junyi_ProblemLog_original.csv"
        )
        logger.info(f"Loading raw data from: {log_pattern}")

        self.sequence_data_lazy = pl.scan_csv(
            log_pattern,
            schema_overrides={
                "user_id": pl.Int64,
                "exercise": pl.Utf8,
                "correct": pl.Utf8,
                "time_done": pl.Int64,
                "count_attempts": pl.Int64,
                "count_hints": pl.Int64,
            },
        ).select(
            [
                "user_id",
                "exercise",
                "correct",
                "time_done",
                "count_attempts",
                "count_hints",
                "time_taken_attempts",
            ]
        )

    def _validate_data_paths(self) -> None:
        """Validate that required data paths exist."""
        if not os.path.exists(self.raw_data_folder):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_folder}")
        logger.info(f"Loading raw data from: {self.raw_data_folder}")

        exercise_path = os.path.join(self.raw_data_folder, "junyi_Exercise_table.csv")
        if not os.path.exists(exercise_path):
            raise FileNotFoundError(f"Cannot find: {exercise_path}")

        log_path = os.path.join(self.raw_data_folder, "junyi_ProblemLog_original.csv")
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Cannot find: {log_path}")

    @override
    def transform_data(self) -> None:
        """Process and clean data."""
        logger.info("Processing Junyi 2015 data...")

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

        # Apply question mapping to cleaned data
        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build normalized relation tables from exercise metadata
        valid_questions = set(self.cleaned_raw_data["question"].unique().to_list())

        exercise_renamed = (
            self.exercise_data.rename(
                {"name": "question", "topic": "skill", "area": "assignment"}
            )
            .select(["question", "skill", "assignment"])
            .filter(pl.col("question").is_in(valid_questions))
            .unique(subset=["question"])
            .with_columns(
                pl.col("question")
                .replace(question_map)
                .cast(pl.Int32)
                .alias("question")
            )
        )
        logger.debug(f"Exercise metadata shape: {exercise_renamed.shape}")

        question_skill = exercise_renamed.select(["question", "skill"])
        question_assignment = exercise_renamed.select(["question", "assignment"])

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

        # Build final sequence_data
        sequence_data = mapped_data.select(
            [
                "user",
                "question",
                "label",
                "attempt_count",
                "hint_count",
                "timestamp",
                "ms_first_response",
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
        }
        self.sequence_data = sequence_data

    @override
    def clean_raw_data(self) -> None:
        """Clean raw sequence data."""
        self.load_src_data()

        sequence_data = (
            self.sequence_data_lazy.filter(
                pl.col("user_id").is_not_null()
                & pl.col("exercise").is_not_null()
                & pl.col("correct").is_not_null()
            )
            .with_columns(
                [
                    # Convert "true"/"false" string to 1/0
                    (pl.col("correct") == "true").cast(pl.Int8).alias("label"),
                    # Convert microseconds to milliseconds
                    (pl.col("time_done") / 1000).cast(pl.Int64).alias("timestamp"),
                    # Extract first attempt time (seconds → milliseconds)
                    (
                        pl.col("time_taken_attempts")
                        .str.split_exact("&", 1)
                        .struct[0]
                        .cast(pl.Float64)
                        * 1000
                    )
                    .cast(pl.Int64)
                    .alias("ms_first_response"),
                ]
            )
            .drop(["correct", "time_done", "time_taken_attempts"])
            .rename(
                {
                    "user_id": "user",
                    "exercise": "question",
                    "count_attempts": "attempt_count",
                    "count_hints": "hint_count",
                }
            )
            .collect(engine="streaming")
        )

        # Convert to global relative time (dataset-wise earliest timestamp as zero)
        sequence_data = sequence_data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        sequence_data = sequence_data.sort(["user", "timestamp"])

        logger.debug(f"Loaded {len(sequence_data)} raw interactions.")

        # Exclude sequences that are too short
        data = exclude_short_sequences(sequence_data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["Junyi2015Data"]
