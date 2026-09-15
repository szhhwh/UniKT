"""NeurIPS 2020 Education Challenge Task 3 & 4 dataset handler."""

import os
from typing import Any

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import DataSource, exclude_short_sequences

logger = get_logger(__name__)


@register_data_source("nips2020_t34")
class NIPS2020T34Data(DataSource):
    """NeurIPS 2020 Education Challenge Task 3 & 4 dataset handler.

    Source: https://eedi.com/projects/neurips-education-challenge
    Uses Level 3 subjects from subject_metadata as skills.
    """

    def __init__(self, args: Any) -> None:
        """Initialize the NIPS 2020 Task 3 & 4 dataset handler."""
        super().__init__(
            dataset="nips2020_t34",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/nips2020_t34.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_dir = os.path.join(self.data_folder, "raw")

    @override
    def load_src_data(self) -> None:
        paths = {
            "primary": os.path.join(self.raw_dir, "train_task_3_4.csv"),
            "answer_meta": os.path.join(self.raw_dir, "answer_metadata_task_3_4.csv"),
            "question_meta": os.path.join(
                self.raw_dir, "question_metadata_task_3_4.csv"
            ),
            "subject_meta": os.path.join(self.raw_dir, "subject_metadata.csv"),
        }
        for name, p in paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Cannot find: {p}")

        logger.info("Loading raw data from: %s", self.raw_dir)
        self.raw_data = {
            "primary": pl.scan_csv(paths["primary"]),
            "answer_meta": pl.scan_csv(paths["answer_meta"]),
            "question_meta": pl.scan_csv(paths["question_meta"]),
            "subject_meta": pl.scan_csv(paths["subject_meta"]),
        }

    @override
    def clean_raw_data(self) -> None:
        if self.raw_data is None:
            self.load_src_data()

        primary = self.raw_data["primary"]
        answer_meta = self.raw_data["answer_meta"]
        subject_meta = self.raw_data["subject_meta"]
        question_meta = self.raw_data["question_meta"]

        # Get Level 3 subject IDs
        level3_subjects = (
            subject_meta.filter(pl.col("Level") == 3)
            .select(pl.col("SubjectId").cast(pl.Int64))
            .collect()
        )
        level3_ids = set(level3_subjects["SubjectId"].to_list())
        logger.info(f"Level 3 subjects: {len(level3_ids)}")

        # Join primary with answer metadata to get timestamp
        data = primary.with_row_index("_row_id").join(
            answer_meta.select(["AnswerId", "DateAnswered"]),
            on="AnswerId",
            how="left",
        )

        data = data.select(
            [
                pl.col("UserId").alias("user"),
                pl.col("QuestionId").alias("question"),
                pl.col("IsCorrect").cast(pl.Int32).alias("label"),
                pl.col("DateAnswered").alias("timestamp"),
                pl.col("_row_id"),
            ]
        )

        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("question").is_not_null()
            & pl.col("label").is_not_null()
            & pl.col("label").is_in([0, 1])
            & pl.col("timestamp").is_not_null()
        )

        data = data.collect(engine="streaming")

        # Parse timestamps to relative milliseconds.
        data = data.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S%.f", strict=False)
            .cast(pl.Int64)
            .alias("timestamp")
        )
        data = data.filter(pl.col("timestamp").is_not_null())

        min_ts = data.select(pl.col("timestamp").min()).item()
        data = data.with_columns(
            (pl.col("timestamp") - min_ts).cast(pl.Int64).alias("timestamp")
        )

        data = data.unique(subset=["user", "question", "timestamp"])
        data = data.sort(["user", "timestamp", "_row_id"])
        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data
        self._level3_ids = level3_ids
        self._question_meta = question_meta

    @override
    def transform_data(self) -> None:
        logger.info("Processing nips2020_t34 data...")

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

        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build question_skill from question metadata, keeping only Level 3 subjects
        question_meta = self._question_meta.collect()
        level3_ids = self._level3_ids

        # Parse SubjectId: "[3, 71, 98]" -> explode rows, filter Level 3
        question_skill = (
            question_meta.select(
                pl.col("QuestionId").alias("question"),
                pl.col("SubjectId")
                .str.strip_chars("[] ")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars().cast(pl.Int64)),
            )
            .explode("SubjectId")
            .rename({"SubjectId": "skill"})
            .filter(pl.col("skill").is_in(level3_ids))
            .with_columns(pl.col("skill").cast(pl.String))
        )

        # Filter to only questions present in sequence data
        active_questions = set(self.cleaned_raw_data["question"].to_list())
        question_skill = question_skill.filter(
            pl.col("question").is_in(active_questions)
        )

        # Apply question mapping to question_skill
        question_skill = (
            question_skill.join(question_map_df, on="question", how="left")
            .filter(pl.col("question_id").is_not_null())
            .drop("question")
            .rename({"question_id": "question"})
            .select(["question", "skill"])
        )
        question_skill = question_skill.unique(subset=["question", "skill"])

        self._build_id_mapping(question_skill, ["skill"])
        question_skill = self._apply_id_mapping(question_skill, columns=["skill"])

        # Filter sequence_data to only questions with skills
        valid_questions = set(question_skill["question"].to_list())
        sequence_data = mapped_data.filter(pl.col("question").is_in(valid_questions))
        sequence_data = sequence_data.select(["user", "question", "label", "timestamp"])

        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])

        self.relation_data = {
            "question_skill": question_skill,
        }
        self.sequence_data = sequence_data


__all__ = ["NIPS2020T34Data"]
