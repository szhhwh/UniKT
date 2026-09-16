"""XES3G5M (question level) dataset handler."""

import os
from typing import Any

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import DataSource, exclude_short_sequences

logger = get_logger(__name__)

_QL_DIR = "question_level"
_TRAIN_FILE = "train_valid_sequences_quelevel.csv"
_TEST_FILE = "test_quelevel.csv"


@register_data_source("xes3g5m")
class Xes3g5mData(DataSource):
    """XES3G5M dataset handler reconstructing question-level interactions.

    The published XES3G5M data is pyKT's processed output. The question_level
    folder packs each student's interactions into 200-length windows (training
    set, with padding flagged by ``selectmasks == -1``) alongside the original,
    un-truncated test sequences. We reconstruct the original question-level
    interaction log by flattening both files to one row per interaction.

    Source files (``data/xes3g5m/raw/question_level/``):
        - ``train_valid_sequences_quelevel.csv``: fold, uid, questions, concepts,
          responses, timestamps, selectmasks. One 200-length window per row.
        - ``test_quelevel.csv``: fold=-1, uid, questions, concepts, responses,
          timestamps. One full sequence per student (no padding).

    Field mapping:
        - uid -> user
        - questions -> question (internal question id)
        - responses -> label
        - timestamps -> timestamp (ms; shifted to be dataset-relative)
        - concepts -> skill; ``_`` joins the KCs of a multi-KC question and each
          KC becomes a row in the question_skill relation.

    The original pyKT fold column is dropped; the framework re-splits
    train/val/test itself. Same-timestamp repeats are preserved as legitimate
    source events (they also occur in the un-windowed test file).
    """

    def __init__(self, args: Any) -> None:
        """Initialize the XES3G5M dataset handler."""
        super().__init__(
            dataset="xes3g5m",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/xes3g5m.zip",
            seed=args.seed,
        )
        self.args = args
        ql = os.path.join(self.raw_folder, _QL_DIR)
        self.train_path = os.path.join(ql, _TRAIN_FILE)
        self.test_path = os.path.join(ql, _TEST_FILE)

    @override
    def load_src_data(self) -> None:
        for path in (self.train_path, self.test_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cannot find: {path}")
        logger.info("Loading XES3G5M question-level data (train_valid + test)")
        self.raw_data = {
            "train": pl.scan_csv(self.train_path),
            "test": pl.scan_csv(self.test_path),
        }

    @staticmethod
    def _flatten(lf: pl.LazyFrame, has_selectmask: bool) -> pl.DataFrame:
        """Explode one packed-sequences file into one row per interaction."""
        list_cols = ["questions", "concepts", "responses", "timestamps"]
        if has_selectmask:
            list_cols.append("selectmasks")

        explode_cols = ["question", "concept", "label", "timestamp"]
        if has_selectmask:
            explode_cols.append("selectmasks")

        lf = (
            lf.with_columns([pl.col(c).str.split(",").alias(c) for c in list_cols])
            .rename(
                {
                    "uid": "user",
                    "questions": "question",
                    "concepts": "concept",
                    "responses": "label",
                    "timestamps": "timestamp",
                }
            )
            .explode(explode_cols)
            .with_columns(
                [
                    pl.col("user").cast(pl.Int32),
                    pl.col("question").cast(pl.Int32),
                    pl.col("label").cast(pl.Int32),
                    pl.col("timestamp").cast(pl.Int64),
                ]
            )
        )
        if has_selectmask:
            lf = lf.filter(pl.col("selectmasks") != "-1").drop("selectmasks")
        # Drop leaked padding (question id -1) and the original pyKT fold.
        lf = lf.filter(pl.col("question") >= 0).drop("fold")
        lf = lf.filter(pl.col("label").is_in([0, 1]))
        return lf.collect(engine="streaming")

    @override
    def clean_raw_data(self) -> None:
        """Flatten both source files into a cleaned long-format interaction table."""
        if self.raw_data is None:
            self.load_src_data()

        train = self._flatten(self.raw_data["train"], has_selectmask=True)
        test = self._flatten(self.raw_data["test"], has_selectmask=False)
        data = pl.concat([train, test], how="vertical")

        # Drop rows with mislabeled timestamps. The interactions form a tight
        # 2020-07..2021-05 cluster, but a handful carry decade-off typos
        # (2001..2018) that would distort the relative-time origin. A 3*IQR
        # fence (Tukey's "far outlier" rule) on the absolute ms timestamp
        # removes exactly these; no-op on clean data.
        q1 = data.select(pl.col("timestamp").quantile(0.25)).item()
        q3 = data.select(pl.col("timestamp").quantile(0.75)).item()
        iqr_fence = 3.0 * (q3 - q1)
        outlier_mask = (pl.col("timestamp") < q1 - iqr_fence) | (
            pl.col("timestamp") > q3 + iqr_fence
        )
        outlier_count = data.filter(outlier_mask).height
        if outlier_count > 0:
            logger.info(
                f"Dropped {outlier_count} interactions with mislabeled "
                "timestamps (outside 3*IQR fence)."
            )
            data = data.filter(~outlier_mask)

        data = data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )
        data = data.sort(["user", "timestamp"])

        data = exclude_short_sequences(data, self.args.min_seq_len)

        logger.debug(
            f"Cleaned data: {data.shape[0]} interactions, "
            f"{data['user'].n_unique()} users, "
            f"{data['question'].n_unique()} questions"
        )
        self.cleaned_raw_data = data

    @override
    def transform_data(self) -> None:
        """Build the question_skill relation and sequence_data from cleaned data."""
        logger.info("Processing XES3G5M data...")
        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # "_" joins the KCs of one question; explode to one row per KC. Each
        # question maps to a single, consistent concept-set in the source data.
        question_skill = (
            self.cleaned_raw_data.select(["question", "concept"])
            .with_columns(pl.col("concept").str.split("_").alias("skill"))
            .explode("skill")
            .with_columns(pl.col("skill").cast(pl.Int32))
            .unique()
        )

        # Dense, contiguous question ids starting at 0.
        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
            .with_columns(pl.col("question_id").cast(pl.Int32))
        )
        self._id_mappings["question"] = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )

        question_skill = (
            question_skill.join(question_map_df, on="question", how="left")
            .drop("question")
            .rename({"question_id": "question"})
        )

        sequence_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
            .select(["user", "question", "label", "timestamp"])
        )

        self._build_id_mapping(question_skill, ["skill"])
        question_skill = self._apply_id_mapping(question_skill, ["skill"])

        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, ["user"])

        logger.debug(
            f"Built question_skill: {question_skill.shape[0]} pairs, "
            f"{self._get_mapped_count('skill')} skills, "
            f"{self._get_mapped_count('user')} users"
        )

        self.relation_data = {
            "question_skill": question_skill.select(["question", "skill"])
        }
        self.sequence_data = sequence_data


__all__ = ["Xes3g5mData"]
