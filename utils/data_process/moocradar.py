"""MOOC-Radar (THU-KEG) fine-grained interaction dataset handler."""

import json
import os
from argparse import Namespace

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
    exclude_short_sequences,
)

logger = get_logger(__name__)


@register_data_source("moocradar")
class MoocRadarData(DataSource):
    """MOOC-Radar fine-grained interaction dataset (THU-KEG).

    Raw files (under ``data/moocradar/raw/``):
        - ``student-problem-fine.json``: JSON array of per-user objects
          ``{"seq": [<interaction>, ...]}``. Each interaction carries
          ``log_id`` (``"{uid}_{n}"``), ``problem_id``, ``user_id``,
          ``is_correct`` (0/1), ``attempts``, ``score``, ``submit_time``
          (``"%Y-%m-%d %H:%M:%S"``).
        - ``problem.json``: JSONL, one record per problem with top-level
          ``problem_id`` and ``concepts`` (list of knowledge-concept strings).

    Mapping to standard format:
        - user_id -> user
        - problem_id -> question
        - is_correct -> label
        - submit_time -> timestamp (ms since the dataset's earliest record)
        - concepts -> skill. A problem maps to MULTIPLE skills, so the
          ``question_skill`` relation is many-to-many -- unlike single-skill
          sets (e.g. Practice Anatomy), it is built from ``problem.json``
          rather than aggregated from the interaction frame.

    Notable data quirks (driven by the actual data, not copied):
        - Records carry no natural order id. Within a user they are ordered by
          ``submit_time``; ~10.8% of adjacent pairs are inverted because
          students submit several problems near-simultaneously. Ties (and the
          inverted runs) are resolved by ``log_seq`` -- the integer tail of
          ``log_id`` -- which is monotonically increasing per user in file
          order (only 5/14224 users violate this) and is the most faithful
          available ordering signal.
        - ``score`` duplicates ``is_correct`` and is null for ~60% of rows;
          ``attempts`` is 1 for 99.98% of rows. Both are dropped.
        - 2513 problems are exercised and all carry >=1 concept (0 missing),
          so every question in the sequences has a skill mapping.
    """

    def __init__(self, args: Namespace) -> None:
        """Initialize the MOOC-Radar dataset handler."""
        super().__init__(
            dataset="moocradar",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/moocradar.zip",
            seed=args.seed,
        )
        self.args = args
        self.fine_path = os.path.join(
            self.data_folder, "raw", "student-problem-fine.json"
        )
        self.problem_path = os.path.join(self.data_folder, "raw", "problem.json")
        # problem_id -> [concept, ...], populated by load_src_data
        self.problem_concepts: dict[str, list[str]] = {}

    @override
    def load_src_data(self) -> None:
        for path in (self.fine_path, self.problem_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cannot find: {path}")

        logger.info(f"Loading problem metadata from: {self.problem_path}")
        self.problem_concepts = {}
        with open(self.problem_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.problem_concepts[obj["problem_id"]] = list(
                    obj.get("concepts") or []
                )

        logger.info(f"Loading fine interactions from: {self.fine_path}")
        users: list[str] = []
        questions: list[str] = []
        labels: list[int] = []
        times: list[str] = []
        log_seqs: list[int] = []
        with open(self.fine_path) as f:
            for obj in json.load(f):
                for rec in obj["seq"]:
                    users.append(rec["user_id"])
                    questions.append(rec["problem_id"])
                    labels.append(rec["is_correct"])
                    times.append(rec["submit_time"])
                    log_seqs.append(int(rec["log_id"].rsplit("_", 1)[-1]))

        self.raw_data = pl.DataFrame(
            {
                "user": users,
                "question": questions,
                "label": labels,
                "timestamp": times,
                "log_seq": log_seqs,
            },
            schema={
                "user": pl.Utf8,
                "question": pl.Utf8,
                "label": pl.Int32,
                "timestamp": pl.Utf8,
                "log_seq": pl.Int64,
            },
        )
        logger.debug(
            f"Loaded {self.raw_data.shape[0]} interactions, "
            f"{self.raw_data['user'].n_unique()} users"
        )

    @override
    def clean_raw_data(self) -> None:
        """Clean raw sequence data."""
        if self.raw_data is None:
            self.load_src_data()

        data = self.raw_data.select(
            ["user", "question", "label", "timestamp", "log_seq"]
        )

        # submit_time "%Y-%m-%d %H:%M:%S" -> epoch ms, then dataset-relative
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
        data = data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        # Order within user by submit_time; log_seq breaks same-timestamp ties.
        data = data.sort(["user", "timestamp", "log_seq"]).drop("log_seq")

        data = data.unique()
        data = data.filter(pl.col("label").is_in([0, 1]))

        data = exclude_short_sequences(data, self.args.min_seq_len)

        logger.debug(
            f"Cleaned data: {data.shape[0]} rows, "
            f"{data['user'].n_unique()} users, "
            f"{data['question'].n_unique()} questions"
        )

        self.cleaned_raw_data = data
        # raw_data is fully consumed; release before the split stages.
        self.raw_data = None

    @override
    def transform_data(self) -> None:
        """Transform cleaned data into standard sequence_data and relations."""
        logger.info("Processing MOOC-Radar data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )
        self._id_mappings["question"] = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )

        # Many-to-many question_skill built from problem concepts (not from the
        # interaction frame, which would collapse multi-skill problems).
        q_to_id = self._id_mappings["question"]
        relation_rows = [
            (q_id, concept)
            for q_str, q_id in q_to_id.items()
            for concept in self.problem_concepts.get(q_str, [])
        ]
        question_skill = pl.DataFrame(
            relation_rows,
            schema={"question": pl.Int32, "skill": pl.Utf8},
        ).unique(subset=["question", "skill"])

        self._build_id_mapping(question_skill, ["skill"])
        question_skill = self._apply_id_mapping(question_skill, ["skill"])

        # One row per interaction; skills live only in the relation table.
        sequence_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
            .select(["user", "question", "label", "timestamp"])
        )

        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])

        logger.debug(
            f"question_skill: {question_skill.shape[0]} pairs, "
            f"{question_skill['skill'].n_unique()} skills, "
            f"{question_skill['question'].n_unique()} questions; "
            f"{sequence_data['user'].n_unique()} users"
        )

        self.relation_data = {"question_skill": question_skill}
        self.sequence_data = sequence_data


__all__ = ["MoocRadarData"]
