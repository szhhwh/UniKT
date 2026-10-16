"""ASSISTments 2009-2010 dataset handler."""

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


@register_data_source("assistments09")
class Assistments2009Data(DataSource):
    """Assistments 2009-2010 dataset handler.

    Dataset source: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
    """

    def __init__(self, args: Any) -> None:
        """Initialize the ASSISTments 2009-2010 dataset handler."""
        super().__init__(
            dataset="assistments09",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments09.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "skill_builder_data_corrected_collapsed.csv"
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
            null_values=["NA"],
        ).lazy()

    @override
    def transform_data(self) -> None:
        """Clean data and build question_data and sequence_data."""
        logger.info("Processing ASSISTments 2009 data...")

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

        # ASSISTments 09 skill format: "2_37_70" -> ["2", "37", "70"]
        skill_data = mapped_data.select(
            pl.col("skill").unique().str.split("_")
        ).explode("skill", empty_as_null=True)

        question_assignment = mapped_data.select(["question", "assignment"]).unique(
            subset=["question", "assignment"]
        )
        logger.debug(f"question_assignment shape: {question_assignment.shape}")

        question_template = mapped_data.select(["question", "template"]).unique(
            subset=["question", "template"]
        )
        logger.debug(f"question_template shape: {question_template.shape}")

        # Build ID mappings for skills and item metadata.
        self._build_id_mapping(skill_data, ["skill"])
        self._build_id_mapping(question_assignment, ["assignment"])
        self._build_id_mapping(question_template, ["template"])
        logger.debug(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}, "
            f"templates={self._get_mapped_count('template')}"
        )

        # Apply ID mappings independently
        question_assignment = self._apply_id_mapping(
            question_assignment, columns=["assignment"]
        )
        question_template = self._apply_id_mapping(
            question_template, columns=["template"]
        )

        # Build final sequence_data
        mapped_data = mapped_data.with_columns(
            self._map_interaction_skills(pl.col("skill").str.split("_"))
        )
        sequence_data = mapped_data.select(
            [
                "interaction_id",
                "skills",
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

        # Store processed data
        self.relation_data = {
            "question_assignment": question_assignment,
            "question_template": question_template,
        }
        self.set_sequence_data(sequence_data)

    def clean_raw_data(self) -> None:
        """Clean raw sequence data."""
        if self.raw_data is None:
            self.load_src_data()

        schema_names = self.raw_data.collect_schema().names()
        data = self.raw_data.with_row_index("interaction_id")
        if "" in schema_names:
            data = data.drop("")

        # Rename columns
        data = data.rename(
            {
                "correct": "label",
                "user_id": "user",
                "problem_id": "question",
                "assignment_id": "assignment",
                "skill_id": "skill",
                "template_id": "template",
                "order_id": "timestamp",
            }
        )

        # Filter invalid data
        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("skill").is_not_null()
            & pl.col("label").is_not_null()
            # Drop clock artifacts (negative / 0 ms) so time features stay valid
            & (pl.col("ms_first_response") > 0)
        )

        data = data.unique(
            subset=pl.exclude("interaction_id"), keep="first", maintain_order=True
        ).collect()

        # Convert to global relative time (dataset-wise earliest timestamp as zero)
        data = data.with_columns(
            pl.col("timestamp")
            .cast(pl.Int64)
            .sub(pl.col("timestamp").cast(pl.Int64).min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        data = data.sort(self._sort_columns())

        # Exclude sequences that are too short
        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["Assistments2009Data"]
