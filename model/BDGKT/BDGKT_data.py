"""BDGKT data preparation module."""

from collections import defaultdict

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.core import get_logger
from utils.model_data import BaseModelData, QuestionModelData
from utils.model_data.packed_sequences import SequencePartition

logger = get_logger(__name__)

__all__ = ["BDGKTModelData", "BDGKTDataset"]


class BDGKTDataset(Dataset):
    """BDGKT target dataset."""

    def __init__(self, data):
        """
        Args:
            data: dict with keys:
                target_students [N], target_questions [N], labels [N],
                target_skills [N, K], hist_skills [N, l_s, K],
                hist_q [N, l_s], hist_r [N, l_s], hist_mask [N, l_s],
                q_ans_s [N, l_s, l_q], q_ans_r [N, l_s, l_q],
                q_ans_mask [N, l_s, l_q]
        """
        self.data = data

    def __len__(self):
        return self.data["target_students"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def _collate_fn(batch):
    """Pack list of sample dicts into batched tensors."""
    keys = [
        "target_students",
        "target_questions",
        "labels",
        "target_skills",
        "hist_skills",
        "hist_q",
        "hist_r",
        "hist_mask",
        "q_ans_s",
        "q_ans_r",
        "q_ans_mask",
    ]
    return tuple(torch.stack([b[k] for b in batch]) for k in keys)


class BDGKTModelData(QuestionModelData):
    """BDGKT model data preparation."""

    @BaseModelData.disk_cache()
    def prepare_data(self, rc):
        l_s = rc.model.question_max_length
        l_q = rc.model.student_max_length

        events = self.data_src.get_sequence_data()
        raw_data = events.to_pandas()
        logger.info(
            f"Loaded {len(raw_data)} interactions, "
            f"{raw_data['user'].nunique()} users, "
            f"{raw_data['question'].nunique()} questions"
        )

        edges = self._build_sorted_edges(raw_data)

        num_students = edges["num_students"]
        membership = (
            events.group_by("user")
            .agg(pl.col("fold").first(), pl.col("fold").n_unique().alias("fold_count"))
            .sort("user")
        )
        if membership.filter(pl.col("fold_count") != 1).height:
            raise ValueError("Each student must belong to one fold")
        partition = SequencePartition.from_folds(
            membership["fold"].to_numpy(),
            rc.data.fold,
            self.data_src.get_metadata("kfold_n_splits"),
        )
        fold_idx = rc.data.fold

        user_ids = membership["user"].to_numpy()
        train_users = user_ids[partition.train]
        val_users = user_ids[partition.val]
        test_users = user_ids[partition.test]

        logger.info(
            f"Fold {fold_idx}: train={len(train_users)}, "
            f"val={len(val_users)}, test={len(test_users)} users"
        )

        student_qr_sets = _build_student_qr_sets(edges, l_s)
        train_ds = _build_dataset(train_users, edges, student_qr_sets, l_s, l_q)
        val_ds = _build_dataset(val_users, edges, student_qr_sets, l_s, l_q)
        test_ds = _build_dataset(test_users, edges, student_qr_sets, l_s, l_q)

        logger.info(
            f"Targets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

        return train_ds, val_ds, test_ds, num_students, edges["num_questions"]

    def _build_sorted_edges(self, raw_data):
        users = raw_data["user"].values.astype(np.int64)
        questions = raw_data["question"].values.astype(np.int64)
        labels = raw_data["label"].values.astype(np.int64)
        timestamps = raw_data["timestamp"].values.astype(np.float64)
        interaction_ids = raw_data["interaction_id"].values.astype(np.int64)
        event_skills = raw_data["skills"].values
        skill_width = max(map(len, event_skills))
        skills = np.full((len(raw_data), skill_width), -1, dtype=np.int64)
        for i, annotated_skills in enumerate(event_skills):
            skills[i, : len(annotated_skills)] = annotated_skills

        num_students = int(users.max()) + 1
        num_questions = int(questions.max()) + 1

        s2q_order = np.lexsort((interaction_ids, timestamps, users))
        s2q_users = users[s2q_order]
        s2q_questions = questions[s2q_order]
        s2q_labels = labels[s2q_order]
        s2q_timestamps = timestamps[s2q_order]

        user_counts = np.bincount(s2q_users, minlength=num_students)
        s_offset = np.zeros(num_students + 1, dtype=np.int64)
        np.cumsum(user_counts, out=s_offset[1:])

        q2s_order = np.lexsort((interaction_ids, timestamps, questions))
        q2s_questions = questions[q2s_order]
        q2s_users = users[q2s_order]
        q2s_labels = labels[q2s_order]
        q2s_timestamps = timestamps[q2s_order]

        question_counts = np.bincount(q2s_questions, minlength=num_questions)
        q_offset = np.zeros(num_questions + 1, dtype=np.int64)
        np.cumsum(question_counts, out=q_offset[1:])

        return {
            "s2q_edge_index": torch.tensor(
                np.stack([s2q_users, s2q_questions]), dtype=torch.long
            ),
            "s2q_response": torch.tensor(s2q_labels, dtype=torch.long),
            "s2q_skills": torch.tensor(skills[s2q_order], dtype=torch.long),
            "s2q_timestamp": torch.tensor(s2q_timestamps, dtype=torch.float),
            "student_offset": torch.tensor(s_offset, dtype=torch.long),
            "student_len": torch.tensor(user_counts, dtype=torch.long),
            "q2s_edge_index": torch.tensor(
                np.stack([q2s_questions, q2s_users]), dtype=torch.long
            ),
            "q2s_response": torch.tensor(q2s_labels, dtype=torch.long),
            "q2s_timestamp": torch.tensor(q2s_timestamps, dtype=torch.float),
            "question_offset": torch.tensor(q_offset, dtype=torch.long),
            "question_len": torch.tensor(question_counts, dtype=torch.long),
            "num_students": num_students,
            "num_questions": num_questions,
        }


def _build_student_qr_sets(edges, l_s):
    """Precompute recent question-response sets for student similarity."""
    s2q_ei = edges["s2q_edge_index"].numpy()
    s2q_r = edges["s2q_response"].numpy()
    s_offset = edges["student_offset"].numpy()
    num_students = edges["num_students"]

    # per-student QR sets (last l_s interactions)
    student_qr_sets = [None] * num_students
    for s in range(num_students):
        start = s_offset[s]
        end = s_offset[s + 1]
        scan_start = max(start, end - l_s)
        qr = set()
        for pos in range(scan_start, end):
            qr.add((int(s2q_ei[1, pos]), int(s2q_r[pos])))
        student_qr_sets[s] = frozenset(qr)

    # per-student question set (for fast intersection check)
    student_q_sets = [frozenset(q for q, _ in qr) for qr in student_qr_sets]

    return {
        "student_qr_sets": student_qr_sets,
        "student_q_sets": student_q_sets,
    }


def _collect_targets(user_indices, s_offset_np, s_len_np):
    """Collect edge indices for all prediction targets."""
    counts = s_len_np[user_indices] - 1
    counts = np.maximum(counts, 0)
    N = int(counts.sum())
    if N == 0:
        return np.array([], np.int64), N

    e_indices = np.empty(N, dtype=np.int64)
    write_pos = 0
    for u in user_indices:
        start = s_offset_np[u]
        length = s_len_np[u]
        for p in range(1, length):
            e_indices[write_pos] = start + p
            write_pos += 1
    return e_indices, N


def _extract_history(
    e_indices, t_students, l_s, s_offset_np, s2q_ei, s2q_r, s2q_t, s2q_skills
):
    """Extract l_s history for all targets.

    Returns:
        hist_q, hist_r, hist_mask, hist_ts [N, l_s], hist_skills [N, l_s, K]
    """
    steps = np.arange(l_s - 1, -1, -1, dtype=np.int64)
    raw_idx = e_indices[:, None] - steps[None, :]  # [N, l_s]
    s_starts = s_offset_np[t_students]

    hist_mask = (raw_idx >= s_starts[:, None]) & (raw_idx < e_indices[:, None])

    safe_idx = np.where(hist_mask, raw_idx, 0)
    hist_q = np.where(hist_mask, s2q_ei[1, safe_idx], 0)
    hist_r = np.where(hist_mask, s2q_r[safe_idx], 0)
    hist_ts = np.where(hist_mask, s2q_t[safe_idx], 0.0)
    hist_skills = np.where(hist_mask[..., None], s2q_skills[safe_idx], -1)

    return hist_q, hist_r, hist_mask, hist_ts, hist_skills


def _build_dataset(user_indices, edges, student_histories, l_s, l_q):
    """Build fixed-shape context tensors for a group of users."""
    s_offset_np = edges["student_offset"].numpy()
    s_len_np = edges["student_len"].numpy()
    s2q_ei = edges["s2q_edge_index"].numpy()
    s2q_r = edges["s2q_response"].numpy()
    s2q_skills = edges["s2q_skills"].numpy()
    s2q_t = edges["s2q_timestamp"].numpy()
    q_offset_np = edges["question_offset"].numpy()
    q2s_ei = edges["q2s_edge_index"].numpy()
    q2s_r = edges["q2s_response"].numpy()
    q2s_t = edges["q2s_timestamp"].numpy()

    student_qr_sets = student_histories["student_qr_sets"]
    student_q_sets = student_histories["student_q_sets"]

    # Phase 1: collect targets
    e_indices, N = _collect_targets(user_indices, s_offset_np, s_len_np)
    if N == 0:
        return BDGKTDataset(_empty_data(l_s, l_q, s2q_skills.shape[-1]))

    t_students = s2q_ei[0, e_indices]
    t_questions = s2q_ei[1, e_indices]
    t_labels = s2q_r[e_indices].copy()
    t_skills = s2q_skills[e_indices]

    # Phase 2: history extraction
    hist_q, hist_r, hist_mask, hist_ts, hist_skills = _extract_history(
        e_indices, t_students, l_s, s_offset_np, s2q_ei, s2q_r, s2q_t, s2q_skills
    )

    # Phase 3: group by target student, reuse Jaccard
    student_to_samples = defaultdict(list)
    for i in range(N):
        student_to_samples[int(t_students[i])].append(i)

    q_ans_s = np.zeros((N, l_s, l_q), dtype=np.int64)
    q_ans_r = np.zeros((N, l_s, l_q), dtype=np.int64)
    q_ans_mask = np.zeros((N, l_s, l_q), dtype=np.bool_)

    for target_s, sample_indices in tqdm(
        student_to_samples.items(), desc="Computing jaccard similarity"
    ):
        # precompute Jaccard with all relevant students
        target_qr = student_qr_sets[target_s]
        target_q_set = set(student_q_sets[target_s])

        # students sharing any question with the target
        relevant = set()
        for q in target_q_set:
            q_s = q_offset_np[q]
            q_e = q_offset_np[q + 1]
            relevant.update(q2s_ei[1, q_s:q_e].tolist())
        relevant.discard(target_s)

        # batch Jaccard computation
        jaccard_map = {}
        for other_s in relevant:
            other_q_set = student_q_sets[other_s]
            shared_q = target_q_set & other_q_set
            if not shared_q:
                continue
            other_qr = student_qr_sets[other_s]
            other_filtered = frozenset((q, r) for q, r in other_qr if q in shared_q)
            intersection = len(target_qr & other_filtered)
            union = len(target_qr | other_filtered)
            if union > 0:
                jaccard_map[other_s] = intersection / union

        # select candidates for each sample of this student
        for i in sample_indices:
            for j in range(l_s):
                if not hist_mask[i, j]:
                    continue

                h_q = int(hist_q[i, j])
                h_t = float(hist_ts[i, j])

                # binary search + numpy slice for candidates
                q_s = int(q_offset_np[h_q])
                q_e = int(q_offset_np[h_q + 1])
                ans_end = q_s + int(np.searchsorted(q2s_t[q_s:q_e], h_t))
                if ans_end <= q_s:
                    continue

                answerers = q2s_ei[1, q_s:ans_end]
                valid = answerers != target_s
                candidates = np.flatnonzero(valid) + q_s
                if len(candidates) == 0:
                    continue
                if len(candidates) > l_q * 3:
                    candidates = candidates[-(l_q * 3) :]

                # sort by Jaccard
                scored = [
                    (int(pos), jaccard_map[int(q2s_ei[1, pos])])
                    for pos in candidates
                    if int(q2s_ei[1, pos]) in jaccard_map
                ]
                if not scored:
                    continue
                scored.sort(key=lambda x: -x[1])

                for rank in range(min(l_q, len(scored))):
                    ans_pos, _ = scored[rank]
                    q_ans_s[i, j, rank] = q2s_ei[1, ans_pos]
                    q_ans_r[i, j, rank] = q2s_r[ans_pos]
                    q_ans_mask[i, j, rank] = True

    data = {
        "target_students": torch.tensor(t_students, dtype=torch.long),
        "target_questions": torch.tensor(t_questions, dtype=torch.long),
        "labels": torch.tensor(t_labels, dtype=torch.long),
        "target_skills": torch.tensor(t_skills, dtype=torch.long),
        "hist_skills": torch.tensor(hist_skills, dtype=torch.long),
        "hist_q": torch.tensor(hist_q, dtype=torch.long),
        "hist_r": torch.tensor(hist_r, dtype=torch.long),
        "hist_mask": torch.tensor(hist_mask, dtype=torch.bool),
        "q_ans_s": torch.tensor(q_ans_s, dtype=torch.long),
        "q_ans_r": torch.tensor(q_ans_r, dtype=torch.long),
        "q_ans_mask": torch.tensor(q_ans_mask, dtype=torch.bool),
    }
    return BDGKTDataset(data)


def _empty_data(l_s, l_q, skill_width):
    return {
        "target_students": torch.zeros(0, dtype=torch.long),
        "target_questions": torch.zeros(0, dtype=torch.long),
        "labels": torch.zeros(0, dtype=torch.long),
        "target_skills": torch.full((0, skill_width), -1, dtype=torch.long),
        "hist_skills": torch.full((0, l_s, skill_width), -1, dtype=torch.long),
        "hist_q": torch.zeros(0, l_s, dtype=torch.long),
        "hist_r": torch.zeros(0, l_s, dtype=torch.long),
        "hist_mask": torch.zeros(0, l_s, dtype=torch.bool),
        "q_ans_s": torch.zeros(0, l_s, l_q, dtype=torch.long),
        "q_ans_r": torch.zeros(0, l_s, l_q, dtype=torch.long),
        "q_ans_mask": torch.zeros(0, l_s, l_q, dtype=torch.bool),
    }
