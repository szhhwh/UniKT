"""Concept-sequence data and domain graphs for DisenKT."""

from itertools import combinations
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from utils.model_data import SkillModelData


class DomainSequences(Dataset):
    def __init__(
        self,
        skills: np.ndarray,
        responses: np.ndarray,
        masks: np.ndarray,
        users: np.ndarray,
    ):
        self.skills = torch.from_numpy(skills.astype(np.int64, copy=False))
        self.responses = torch.from_numpy(responses.astype(np.int64, copy=False))
        self.masks = torch.from_numpy(masks.astype(np.bool_, copy=False))
        self.users = users

    def __len__(self) -> int:
        return self.skills.size(0)

    def __getitem__(self, index: int):
        return self.skills[index], self.responses[index], self.masks[index]


def _normalized_concept_graph(
    skills: np.ndarray, masks: np.ndarray, num_skills: int
) -> torch.Tensor:
    """Binary same-sequence co-occurrence with symmetric GCN normalization."""
    edges = {(i, i) for i in range(1, num_skills + 1)}
    for row, mask in zip(skills, masks, strict=True):
        present = np.unique(row[mask])
        for a, b in combinations(present.tolist(), 2):
            edges.add((a, b))
            edges.add((b, a))

    indices = np.asarray(sorted(edges), dtype=np.int64)
    degree = np.bincount(indices[:, 0], minlength=num_skills + 1)
    weights = 1.0 / np.sqrt(degree[indices[:, 0]] * degree[indices[:, 1]])
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices.T.copy()),
        torch.from_numpy(weights.astype(np.float32)),
        (num_skills + 1, num_skills + 1),
    ).coalesce()


class DisenKTModelData(SkillModelData):
    """Use the framework's skill-level split and student-disjoint folds."""

    @staticmethod
    def _rechunk_course_sequences(data: pl.DataFrame, max_len: int) -> pl.DataFrame:
        """Restore continuous per-course order after the framework's mixed-course split."""
        data = data.sort(["user", "sequence_id", "seq_pos"]).with_columns(
            pl.int_range(pl.len()).over(["user", "_domain"]).alias("_domain_pos")
        )
        data = data.with_columns(
            (pl.col("_domain_pos") // max_len).alias("_chunk"),
            (pl.col("_domain_pos") % max_len).alias("seq_pos"),
        )
        sequence_ids = (
            data.select("user", "_domain", "_chunk")
            .unique()
            .sort("user", "_domain", "_chunk")
            .with_row_index("_new_sequence_id")
        )
        return (
            data.join(sequence_ids, on=["user", "_domain", "_chunk"])
            .with_columns(pl.col("_new_sequence_id").alias("sequence_id"))
            .drop("_domain_pos", "_chunk", "_new_sequence_id")
        )

    def _mark_domains(
        self,
        data: pl.DataFrame,
        mode: Literal["student", "course"],
        source_domain: str | None,
        source_domains: list[str] | None,
        target_domain: str | None,
    ) -> pl.DataFrame:
        if mode == "course":
            if "domain" not in data.columns:
                raise ValueError(
                    "DisenKT course-level data requires the 'domain' field."
                )
            if (source_domain is None and not source_domains) or target_domain is None:
                raise ValueError(
                    "DisenKT course-level data requires source_domain or source_domains, and target_domain."
                )
            return data.with_columns(pl.col("domain").cast(pl.String).alias("_domain"))

        interactions = self.data_src.get_sequence_data()
        if "user" not in interactions.columns:
            raise ValueError("DisenKT student-level data requires the 'user' field.")
        counts = interactions.group_by("user").len()
        median = counts["len"].median()
        source_users = counts.filter(pl.col("len") > median)["user"].to_list()
        target_users = counts.filter(pl.col("len") < median)["user"].to_list()
        return data.filter(
            pl.col("user").is_in(source_users + target_users)
        ).with_columns(
            pl.when(pl.col("user").is_in(source_users))
            .then(pl.lit("source"))
            .otherwise(pl.lit("target"))
            .alias("_domain")
        )

    @staticmethod
    def _sequences(data: pl.DataFrame, max_len: int, min_len: int):
        data = data.filter(pl.len().over("sequence_id") >= min_len)
        if data.is_empty():
            raise ValueError("DisenKT requires sequences in every selected domain.")
        data = data.sort(["sequence_id", "seq_pos"])
        ids = data["sequence_id"].to_numpy()
        _, row_index = np.unique(ids, return_inverse=True)
        start = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        positions = np.arange(len(ids)) - np.repeat(
            start, np.diff(np.r_[start, len(ids)])
        )
        n_sequences = len(start)
        skills = np.zeros((n_sequences, max_len), dtype=np.int64)
        responses = np.zeros_like(skills)
        masks = np.zeros((n_sequences, max_len), dtype=np.bool_)
        skills[row_index, positions] = data["skill"].to_numpy() + 1
        responses[row_index, positions] = data["label"].to_numpy()
        masks[row_index, positions] = True
        folds = data["fold"].to_numpy()[start]
        users = data["user"].to_numpy()[start]
        return skills, responses, masks, folds, users

    def prepare_data(self, rc):
        data = self.data_src.get_split_skill_sequence_data()
        required = {"sequence_id", "seq_pos", "user", "skill", "label", "fold"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"DisenKT requires skill sequence fields: {sorted(missing)}"
            )

        m = rc.model
        data = self._mark_domains(
            data,
            m.domain_mode,
            m.source_domain,
            m.source_domains,
            m.target_domain,
        )
        if m.domain_mode == "student":
            selected_domains = {"source": "source", "target": "target"}
        else:
            if m.source_domains:
                source_names = list(m.source_domains)
            elif m.source_domain == "all":
                source_names = [
                    name
                    for name in ("C", "C++", "Java", "Python")
                    if name != m.target_domain
                ]
            else:
                source_names = [m.source_domain]
            source_names = [name for name in source_names if name != m.target_domain]
            selected_domains = {
                f"source_{index}": name for index, name in enumerate(source_names)
            }
            selected_domains["target"] = m.target_domain
        max_len = self.data_src.get_metadata("max_seq_len")
        if m.domain_mode == "course":
            data = data.filter(pl.col("_domain").is_in(list(selected_domains.values())))
            # PTADisc retains students with at least ten interactions per course.
            data = data.filter(pl.len().over(["user", "_domain"]) >= 10)
            common_users = set(data["user"].to_list())
            for name in selected_domains.values():
                common_users &= set(
                    data.filter(pl.col("_domain") == name)["user"].to_list()
                )
            data = data.filter(pl.col("user").is_in(common_users))
            data = self._rechunk_course_sequences(data, max_len)
            data = data.filter(pl.len().over("sequence_id") >= rc.data.min_seq_len)
            common_users = set(data["user"].to_list())
            for name in selected_domains.values():
                common_users &= set(
                    data.filter(pl.col("_domain") == name)["user"].to_list()
                )
            data = data.filter(pl.col("user").is_in(common_users))
        num_skills = self.data_src.get_metadata("num_skills")
        fold = rc.data.fold

        domains = {}
        graphs = {}
        for name, value in selected_domains.items():
            sequences = self._sequences(
                data.filter(pl.col("_domain") == value), max_len, rc.data.min_seq_len
            )
            skills, responses, masks, folds, users = sequences
            train = (folds != fold) & (folds != -1)
            valid = folds == fold
            test = folds == -1
            domains[name] = (
                DomainSequences(
                    skills[train], responses[train], masks[train], users[train]
                ),
                DomainSequences(
                    skills[valid], responses[valid], masks[valid], users[valid]
                ),
                DomainSequences(
                    skills[test], responses[test], masks[test], users[test]
                ),
            )
            graphs[name] = _normalized_concept_graph(
                skills[train], masks[train], num_skills
            )

        train_domains = {name: splits[0] for name, splits in domains.items()}
        if any(len(dataset) == 0 for dataset in train_domains.values()):
            raise ValueError("DisenKT requires training sequences in every domain.")
        return (
            train_domains,
            domains["target"][1],
            domains["target"][2],
            graphs,
            num_skills,
            max_len,
        )
