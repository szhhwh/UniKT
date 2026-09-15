"""Practice Anatomy (practiceanatomy.com) dataset handler."""

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


@register_data_source("practiceanatomy")
class PracticeAnatomyData(DataSource):
    """Practice Anatomy (practiceanatomy.com) dataset handler.

    Students practice locating anatomical structures on images. Each item can
    be asked in two directions, so the same underlying term yields two
    questions, mirroring the Slepemapy structure.

    Raw data columns (answers.csv, comma-separated):
        - id: Answer identifier
        - user: Student ID
        - item_asked: Identifier of the asked term (the underlying concept)
        - term_name_asked: Name of the asked term
        - item_answered: Identifier of the answered term (empty on "I don't know")
        - term_name_answered: Name of the answered term
        - context_name: Image / context name
        - type: Answer direction: t2d (find term on image) or d2t (name highlighted term)
        - options: Number of options
        - time: Datetime the answer was inserted
        - response_time: Response time in milliseconds
        - lang: Terminology language
        - locations_asked / systems_asked / locations_answered / systems_answered: JSON metadata
        - practice_filter: Filter used for practice (JSON)
        - ip_country / ip_id: IP-derived fields

    Mapping to standard format:
        - user → user
        - f"{item_asked}_{type}" → question (asked term + answer direction)
        - item_asked → skill
        - item_asked == item_answered → label (1=correct, 0=incorrect;
          "I don't know" with null item_answered is treated as incorrect)
        - type → assignment (string t2d/d2t, mapped to dense ints downstream)
        - time → timestamp

    Notable differences from Slepemapy (driven by the actual data, not copied):
        - Comma delimiter (Slepemapy uses ';').
        - type is a string ("t2d"/"d2t"); it is kept as a string here and the
          framework's ID mapping converts it to dense ints in transform_data.
        - Timestamps carry microsecond precision ("%Y-%m-%d %H:%M:%S%.f").
        - No null-filter on the answered term: in this release there are zero
          empty item_answered fields, and an absent answer is an explicit
          "I don't know" (incorrect), so such rows are retained with label 0
          rather than dropped.
    """

    def __init__(self, args: Any) -> None:
        """Initialize the Practice Anatomy dataset handler."""
        super().__init__(
            dataset="practiceanatomy",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/practiceanatomy.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(self.data_folder, "raw", "answers.csv")

    @override
    def load_src_data(self) -> None:
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = pl.scan_csv(
            self.raw_data_path,
            separator=",",
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
                pl.col("item_asked"),
                pl.col("item_answered"),
                pl.col("type"),
                pl.col("time").alias("timestamp"),
            ]
        )

        # Build label: correct if asked term matches answered term. An absent
        # answer ("I don't know") compares to null, so fill to incorrect.
        data = data.with_columns(
            (pl.col("item_asked") == pl.col("item_answered"))
            .fill_null(False)
            .cast(pl.Int32)
            .alias("label")
        )

        # Build question from asked item + answer direction combination
        data = data.with_columns(
            (
                pl.col("item_asked").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("type").cast(pl.Utf8)
            )
            .alias("question")
            .cast(pl.Categorical)
        )

        # Drop item_answered, then rename remaining columns. type stays a
        # string (t2d/d2t) until transform_data maps it to dense ints.
        data = data.drop("item_answered")
        data = data.rename({"item_asked": "skill", "type": "assignment"})

        data = data.collect(engine="streaming")

        # Convert timestamp string (microsecond precision) to milliseconds
        data = data.with_columns(
            (
                pl.col("timestamp")
                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
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
        logger.info("Processing Practice Anatomy data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping (question = "item_asked_type" string)
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

        # Build ID mappings from each relation (assignment string -> dense int)
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


__all__ = ["PracticeAnatomyData"]
