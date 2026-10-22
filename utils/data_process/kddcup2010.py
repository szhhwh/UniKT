"""KDD Cup 2010 datasets (Algebra 2005/2006, Bridge to Algebra 2006)."""

import os
from typing import Any

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import DataSource, exclude_short_sequences

logger = get_logger(__name__)


class KDDCup2010Base(DataSource):
    """Base class for KDD Cup 2010 datasets (Algebra 2005/2006, Bridge 2006).

    These datasets share the same tab-separated format with columns:
    Row, Anon Student Id, Problem Hierarchy, Problem Name, Problem View,
    Step Name, ..., First Transaction Time, ..., Correct First Attempt,
    ..., KC(Default) or KC(SubSkills), Opportunity(...)

    A question is a Problem Name + Step Name pair within each dataset.
    Repeated observations retain independent interaction IDs.
    """

    skill_column: str  # Subclass must set: e.g. "KC(Default)" or "KC(SubSkills)"
    question_columns: tuple[str, ...] = ("Problem Name", "Step Name")

    def __init__(self, args: Any, dataset: str, raw_filename: str) -> None:
        """Initialize the KDD Cup 2010 base dataset handler."""
        super().__init__(
            dataset=dataset,
            data_base_path=args.data_base_path,
            data_url=f"http://cdn.lionhao.top/KTDataset/{dataset}.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(self.data_folder, "raw", raw_filename)

    def _question_expression(self) -> pl.Expr:
        """Escape and join problem and step names into a question key."""
        return pl.concat_str(
            [
                pl.col(column)
                .str.replace_all("_", "####", literal=True)
                .str.replace_all(",", "@@@@", literal=True)
                for column in self.question_columns
            ],
            separator="----",
        )

    @override
    def load_src_data(self) -> None:
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = pl.scan_csv(
            self.raw_data_path,
            separator="\t",
            schema_overrides=dict.fromkeys(self.question_columns, pl.String),
            ignore_errors=True,
            try_parse_dates=False,
            null_values=[""],
        )

    @override
    def clean_raw_data(self) -> None:
        if self.raw_data is None:
            self.load_src_data()

        skill_col = self.skill_column
        required_columns = [
            "Anon Student Id",
            *self.question_columns,
            "First Transaction Time",
            "Correct First Attempt",
            skill_col,
        ]
        source_columns = self.raw_data.collect_schema().names()
        missing_columns = [col for col in required_columns if col not in source_columns]
        if missing_columns:
            raise ValueError(
                f"{self.dataset} raw data is missing columns: {missing_columns}"
            )

        data = self.raw_data.select(
            [
                pl.col("Anon Student Id").alias("user"),
                *[pl.col(column) for column in self.question_columns],
                self._question_expression().alias("question"),
                pl.col("First Transaction Time").alias("raw_timestamp"),
                pl.col("Correct First Attempt").alias("raw_label"),
                pl.col(skill_col).alias("skill"),
            ]
        ).with_row_index("row_idx")

        data = data.with_columns(
            [
                pl.col("raw_label").cast(pl.Int32, strict=False).alias("label"),
                pl.col("raw_timestamp")
                .str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S%.f", strict=False)
                .alias("timestamp"),
            ]
        ).collect(engine="streaming")

        invalid_label_count = data.filter(
            pl.col("raw_label").is_not_null()
            & (pl.col("label").is_null() | ~pl.col("label").is_in([0, 1]))
        ).height
        if invalid_label_count > 0:
            raise ValueError(
                f"{self.dataset} contains {invalid_label_count} invalid labels; "
                "expected Correct First Attempt to be 0 or 1."
            )

        invalid_timestamp_count = data.filter(
            pl.col("raw_timestamp").is_not_null() & pl.col("timestamp").is_null()
        ).height
        if invalid_timestamp_count > 0:
            raise ValueError(
                f"{self.dataset} contains {invalid_timestamp_count} unparsable "
                "First Transaction Time values."
            )

        before_filter_count = data.height

        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("question").is_not_null()
            & pl.col("timestamp").is_not_null()
            & pl.col("label").is_not_null()
            & pl.col("skill").is_not_null()
        )

        dropped_count = before_filter_count - data.height
        if dropped_count > 0:
            logger.info(
                f"Dropped {dropped_count} {self.dataset} rows with missing "
                "user/question/timestamp/label/skill."
            )

        # This text encoding is not injective for arbitrary field contents
        # (e.g. literal "####" versus "_"). Never silently merge distinct pairs.
        question_pairs = data.select("question", *self.question_columns).unique()
        if question_pairs["question"].n_unique() != question_pairs.height:
            raise ValueError(
                f"{self.dataset} has distinct Problem Name + Step Name pairs "
                "that produce the same question key."
            )
        data = data.drop(self.question_columns)

        min_ts = data.select(pl.col("timestamp").min()).item()
        if min_ts is None:
            raise ValueError(f"{self.dataset} has no valid rows after cleaning.")

        data = data.with_columns(
            pl.col("timestamp").dt.timestamp("ms").alias("timestamp")
        )

        # Drop rows with mislabeled First Transaction Time. The KDD school-year
        # data forms a tight timestamp cluster, but a handful of rows carry
        # decade-off typos (e.g. algebra2006 has rows dated 1993/2014/2015
        # among a 2006-2007 bulk) that would otherwise inflate DKTForget's
        # gap vocab and distort min-relative time. A 3*IQR fence (Tukey's
        # "far outlier" rule) on the absolute ms timestamp removes exactly
        # these without touching legitimate rows; no-op on clean datasets.
        q1 = data.select(pl.col("timestamp").quantile(0.25)).item()
        q3 = data.select(pl.col("timestamp").quantile(0.75)).item()
        iqr_fence = 3.0 * (q3 - q1)
        outlier_mask = (pl.col("timestamp") < q1 - iqr_fence) | (
            pl.col("timestamp") > q3 + iqr_fence
        )
        outlier_count = data.filter(outlier_mask).height
        if outlier_count > 0:
            logger.info(
                f"Dropped {outlier_count} {self.dataset} rows with mislabeled "
                "First Transaction Time (outside 3*IQR fence)."
            )
            data = data.filter(~outlier_mask)

        min_ts = data.select(pl.col("timestamp").min()).item()
        data = data.with_columns(
            (pl.col("timestamp") - min_ts).alias("timestamp")
        ).drop(["raw_timestamp", "raw_label"])

        data = data.sort(["user", "timestamp", "row_idx"]).rename(
            {"row_idx": "interaction_id"}
        )

        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data

    @override
    def transform_data(self) -> None:
        logger.info(f"Processing {self.dataset} data...")

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

        skill_data = mapped_data.select(
            pl.col("skill").unique().str.split("~~")
        ).explode("skill", empty_as_null=True)
        self._build_id_mapping(skill_data, ["skill"])

        # Keep the observed annotations separate from the question-level relation.
        sequence_data = mapped_data.select(
            [
                "user",
                "question",
                "label",
                "timestamp",
                pl.col("interaction_id").cast(pl.Int64),
                self._map_interaction_skills(pl.col("skill").str.split("~~")),
            ]
        )

        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])

        self.set_sequence_data(sequence_data)


@register_data_source("algebra2005")
class Algebra2005Data(KDDCup2010Base):
    """Algebra 2005-2006 dataset handler.

    Uses KC(Default) and the shared Problem Name + Step Name question identity.
    """

    skill_column = "KC(Default)"

    def __init__(self, args: Any) -> None:
        """Initialize the Algebra 2005-2006 dataset handler."""
        super().__init__(
            args=args,
            dataset="algebra2005",
            raw_filename="algebra_2005_2006_train.txt",
        )


@register_data_source("algebra2006")
class Algebra2006Data(KDDCup2010Base):
    """Algebra 2006-2007 dataset handler.

    Uses KC(Default) and the shared Problem Name + Step Name question identity.
    """

    skill_column = "KC(Default)"

    def __init__(self, args: Any) -> None:
        """Initialize the Algebra 2006-2007 dataset handler."""
        super().__init__(
            args=args,
            dataset="algebra2006",
            raw_filename="algebra_2006_2007_train.txt",
        )


@register_data_source("bridge2006")
class Bridge2006Data(KDDCup2010Base):
    """Bridge to Algebra 2006-2007 dataset handler.

    Uses KC(SubSkills) and the shared Problem Name + Step Name question identity.
    """

    skill_column = "KC(SubSkills)"

    def __init__(self, args: Any) -> None:
        """Initialize the Bridge to Algebra 2006-2007 dataset handler."""
        super().__init__(
            args=args,
            dataset="bridge2006",
            raw_filename="bridge_to_algebra_2006_2007_train.txt",
        )
