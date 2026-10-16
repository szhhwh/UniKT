from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


# ===== Configuration =====
@dataclass
class DyGKTNeighborConfig:
    """DyGKT neighbor-sampling configuration (data-layer helper)."""

    num_question: int
    num_neighbor: int = 50
    sampling_strategy: str = "time_decay"
    time_decay_factor: float = 1e-5
    candidate_pool: int = 200
    seed: int = 2020

    @classmethod
    def from_run_config(cls, rc: Any, num_question: int) -> "DyGKTNeighborConfig":
        m = rc.model
        return cls(
            num_question=num_question,
            num_neighbor=int(m.num_neighbor),
            sampling_strategy=str(m.neighbor_sampling_strategy).lower(),
            time_decay_factor=float(m.time_decay_factor),
            candidate_pool=int(m.neighbor_candidate_pool),
            seed=int(m.neighbor_sampling_seed),
        )


# ===== Independent Functions =====
def sample_histories(
    all_history_indices: list[list[int]],
    all_history_times: list[list[int]],
    all_current_times: list[int],
    config: DyGKTNeighborConfig,
) -> list[list[int]]:
    """Sample histories using time-decay or recent strategy."""
    n_samples = len(all_history_indices)
    if n_samples == 0:
        return []

    num_neighbor = config.num_neighbor
    rng = np.random.default_rng(config.seed)

    # Compute lengths once
    hist_lengths = np.array([len(h) for h in all_history_indices], dtype=np.int32)

    # Recent strategy: simple truncation
    if config.sampling_strategy == "recent":
        return [
            h[-num_neighbor:] if length > num_neighbor else list(h)
            for h, length in zip(all_history_indices, hist_lengths)
        ]

    # Determine candidate pool sizes
    candidate_pool = (
        config.candidate_pool if config.candidate_pool > 0 else hist_lengths.max()
    )
    effective_lengths = np.minimum(hist_lengths, candidate_pool)

    # Find samples that need sampling (length > num_neighbor)
    needs_sampling = hist_lengths > num_neighbor

    # Pre-allocate results
    results: list[list[int]] = [None] * n_samples

    # Handle short histories directly
    short_mask = ~needs_sampling
    for i in np.where(short_mask)[0]:
        results[i] = list(all_history_indices[i]) if hist_lengths[i] > 0 else []

    if not needs_sampling.any():
        return results

    # Process samples needing sampling in batches
    sample_indices = np.where(needs_sampling)[0]
    batch_size = 5000

    for batch_start in range(0, len(sample_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(sample_indices))
        batch_sample_ids = sample_indices[batch_start:batch_end]
        batch_n = len(batch_sample_ids)

        # Get max pool for this batch
        batch_effective = effective_lengths[batch_sample_ids]
        max_pool = batch_effective.max()

        # Build matrices
        mat_times = np.zeros((batch_n, max_pool), dtype=np.float64)
        mat_indices = np.zeros((batch_n, max_pool), dtype=np.int32)
        mat_mask = np.zeros((batch_n, max_pool), dtype=bool)

        for idx, (sid, eff_len) in enumerate(zip(batch_sample_ids, batch_effective)):
            if eff_len == 0:
                continue
            h = all_history_indices[sid]
            t = all_history_times[sid]
            start_pos = len(h) - eff_len
            mat_times[idx, :eff_len] = t[start_pos:]
            mat_indices[idx, :eff_len] = h[start_pos:]
            mat_mask[idx, :eff_len] = True

        # Time-decay weights
        batch_current = np.array(
            [all_current_times[s] for s in batch_sample_ids], dtype=np.float64
        )
        deltas = np.maximum(0.0, batch_current[:, None] - mat_times)
        log_weights = np.where(mat_mask, -config.time_decay_factor * deltas, -np.inf)

        # Numerical stability
        log_max = np.where(
            mat_mask.any(axis=1, keepdims=True),
            np.maximum(log_weights.max(axis=1, keepdims=True), -1e9),
            0.0,
        )
        log_weights = np.where(mat_mask, log_weights - log_max, -np.inf)

        # Gumbel-Top-K sampling
        gumbel_noise = -np.log(-np.log(rng.uniform(size=(batch_n, max_pool))))
        keys = np.where(mat_mask, log_weights + gumbel_noise, -np.inf)

        # Select top-K
        topk_pos = np.argpartition(keys, -num_neighbor, axis=1)[:, -num_neighbor:]

        # Build results
        for idx, sid in enumerate(batch_sample_ids):
            pos_mask = mat_mask[idx, topk_pos[idx]]
            if pos_mask.any():
                selected = np.sort(topk_pos[idx, pos_mask])
                results[sid] = [int(mat_indices[idx, p]) for p in selected]
            else:
                results[sid] = all_history_indices[sid][-num_neighbor:]

    return results


def build_tensors(
    data_all: list[dict[str, Any]],
    config: DyGKTNeighborConfig,
    history_data: list[dict[str, Any]] | None = None,
    target_user_ids: set[int] | None = None,
    target_global_ids: set[int] | None = None,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Build interaction tensors from user sequences.

    Args:
        data_all: User sequences with first-skill IDs in skill_seq (one-based,
            with zero reserved for padding).
        config: DyGKT configuration.
        history_data: Additional history data (for val/test sets).
        target_user_ids: Target user IDs to filter (from data_all).
        target_global_ids: Target global interaction IDs to filter.

    Returns:
        Tuple of (tensors dict, target_indices array).
    """
    num_question = config.num_question
    num_neighbor = config.num_neighbor

    # Combine history data if provided
    if history_data is not None:
        all_records = history_data + data_all
        n_history = sum(int(u["seq_len"]) for u in history_data)
    else:
        all_records = data_all
        n_history = 0

    total_interactions = sum(int(u["seq_len"]) for u in all_records)
    n_target_interactions = total_interactions - n_history

    logger.info(
        "DyGKT: building tensors (total=%d, history=%d, target=%d)...",
        total_interactions,
        n_history,
        n_target_interactions,
    )

    # Pre-allocate arrays with optimized dtypes
    idx_arr = np.zeros(total_interactions, dtype=np.int32)
    global_id_arr = np.zeros(total_interactions, dtype=np.int32)
    user_arr = np.zeros(total_interactions, dtype=np.int32)
    question_arr = np.zeros(total_interactions, dtype=np.int32)
    skill_arr = np.zeros(total_interactions, dtype=np.int32)
    time_arr = np.zeros(total_interactions, dtype=np.int64)
    correctness_arr = np.zeros(total_interactions, dtype=np.int8)

    # Phase 1: Build interaction records
    user_hist_indices: list[list[int]] = []
    user_hist_times: list[list[int]] = []
    user_hist_current_times: list[int] = []

    que_his_seqs: dict[int, list[tuple[int, int]]] = {}
    n = 1
    history_limit = (
        max(num_neighbor, config.candidate_pool) if config.candidate_pool > 0 else 0
    )

    for user_data in all_records:
        user_id = num_question + 1 + int(user_data["user_id"])
        seq_len = int(user_data["seq_len"])
        question_seq = user_data["question_seq"][:seq_len]
        skill_seq = user_data["skill_seq"][:seq_len]
        correctness_seq = user_data["correctness_seq"][:seq_len]
        time_seq = user_data["time_seq"][:seq_len]
        interaction_id_seq = user_data.get("interaction_id_seq")
        if interaction_id_seq is not None:
            interaction_id_seq = interaction_id_seq[:seq_len]

        user_history_indices: list[int] = []
        user_history_times: list[int] = []

        for i in range(seq_len):
            pos = n - 1
            q_id = int(question_seq[i])
            t = int(time_seq[i])
            c = int(correctness_seq[i])
            gid = int(interaction_id_seq[i]) if interaction_id_seq else n

            if q_id not in que_his_seqs:
                que_his_seqs[q_id] = []
            que_his_seqs[q_id].append((n, t))

            idx_arr[pos] = n
            global_id_arr[pos] = gid
            user_arr[pos] = user_id
            question_arr[pos] = q_id + 1
            skill_arr[pos] = int(skill_seq[i])
            time_arr[pos] = t
            correctness_arr[pos] = c

            user_hist_indices.append(list(user_history_indices))
            user_hist_times.append(list(user_history_times))
            user_hist_current_times.append(t)

            user_history_indices.append(n)
            user_history_times.append(t)

            if history_limit > 0 and len(user_history_indices) > history_limit:
                user_history_indices = user_history_indices[-history_limit:]
                user_history_times = user_history_times[-history_limit:]

            n += 1

    # Phase 2: Sample user histories
    logger.info("DyGKT: sampling user histories...")
    user_his_sampled = sample_histories(
        user_hist_indices, user_hist_times, user_hist_current_times, config
    )

    # Phase 3: Build question histories
    logger.info("DyGKT: building question histories...")
    histories_by_idx: list[list[int]] = [[] for _ in range(n)]
    que_hist_states: list[dict] = []

    for seq_list in que_his_seqs.values():
        if not seq_list:
            continue
        seq_sorted = sorted(seq_list, key=lambda x: x[1])
        question_history_indices: list[int] = []
        question_history_times: list[int] = []
        i = 0
        seq_len = len(seq_sorted)

        while i < seq_len:
            t = seq_sorted[i][1]
            j = i
            while j < seq_len and seq_sorted[j][1] == t:
                j += 1

            que_hist_states.append(
                {
                    "history_indices": list(question_history_indices),
                    "history_times": list(question_history_times),
                    "current_time": t,
                    "interaction_indices": [seq_sorted[k][0] for k in range(i, j)],
                }
            )

            for k in range(i, j):
                question_history_indices.append(seq_sorted[k][0])
                question_history_times.append(seq_sorted[k][1])

            if history_limit > 0 and len(question_history_indices) > history_limit:
                question_history_indices = question_history_indices[-history_limit:]
                question_history_times = question_history_times[-history_limit:]
            i = j

    # Phase 4: Sample question histories
    if que_hist_states:
        logger.info("DyGKT: sampling question histories...")
        que_sampled = sample_histories(
            [s["history_indices"] for s in que_hist_states],
            [s["history_times"] for s in que_hist_states],
            [s["current_time"] for s in que_hist_states],
            config,
        )
        for state, sampled in zip(que_hist_states, que_sampled):
            for idx in state["interaction_indices"]:
                histories_by_idx[idx] = sampled

    # Phase 5: Build final arrays
    user_his_padded = np.zeros((total_interactions, num_neighbor), dtype=np.int32)
    que_his_padded = np.zeros((total_interactions, num_neighbor), dtype=np.int32)

    for i in range(total_interactions):
        user_seq = user_his_sampled[i]
        if user_seq:
            clip = user_seq[-num_neighbor:]
            user_his_padded[i, : len(clip)] = clip

        interaction_idx = int(idx_arr[i])
        que_seq = histories_by_idx[interaction_idx]
        if que_seq:
            clip = que_seq[-num_neighbor:]
            que_his_padded[i, : len(clip)] = clip

    # Compute lengths
    user_his_len = np.sum(user_his_padded != 0, axis=1).astype(np.int32)
    que_his_len = np.sum(que_his_padded != 0, axis=1).astype(np.int32)

    # Phase 6: Build lookup arrays
    max_idx = int(idx_arr.max()) if total_interactions > 0 else 0
    lookup_user = np.zeros(max_idx + 1, dtype=np.int32)
    lookup_question = np.zeros(max_idx + 1, dtype=np.int32)
    lookup_skill = np.zeros(max_idx + 1, dtype=np.int32)
    lookup_time = np.zeros(max_idx + 1, dtype=np.int64)
    lookup_correctness = np.zeros(max_idx + 1, dtype=np.int8)

    lookup_user[idx_arr] = user_arr
    lookup_question[idx_arr] = question_arr
    lookup_skill[idx_arr] = skill_arr
    lookup_time[idx_arr] = time_arr
    lookup_correctness[idx_arr] = correctness_arr

    # Phase 7: Determine target indices
    # When history_data is provided, target only the data_all portion
    if history_data is not None:
        # Target only the last n_target_interactions
        target_positions = np.arange(n_history, total_interactions)

        # Further filter by user_ids if provided
        if target_user_ids is not None:
            target_mask = np.isin(user_arr[target_positions], list(target_user_ids))
            target_positions = target_positions[target_mask]
    elif target_global_ids is not None:
        target_mask = np.isin(global_id_arr, list(target_global_ids))
        target_positions = np.where(target_mask)[0]
    elif target_user_ids is not None:
        target_mask = np.isin(user_arr, list(target_user_ids))
        target_positions = np.where(target_mask)[0]
    else:
        target_positions = np.arange(total_interactions)

    # Phase 8: Convert to tensors
    tensors = {
        "idx": torch.from_numpy(idx_arr),
        "user": torch.from_numpy(user_arr),
        "question": torch.from_numpy(question_arr),
        "skill": torch.from_numpy(skill_arr),
        "time": torch.from_numpy(time_arr).float(),
        "correctness": torch.from_numpy(correctness_arr).float(),
        "user_his_idx": torch.from_numpy(user_his_padded),
        "user_his_len": torch.from_numpy(user_his_len),
        "que_his_idx": torch.from_numpy(que_his_padded),
        "que_his_len": torch.from_numpy(que_his_len),
        "lookup_user": torch.from_numpy(lookup_user),
        "lookup_question": torch.from_numpy(lookup_question),
        "lookup_skill": torch.from_numpy(lookup_skill),
        "lookup_time": torch.from_numpy(lookup_time).float(),
        "lookup_correctness": torch.from_numpy(lookup_correctness).float(),
    }

    logger.info(
        "DyGKT dataset: total=%d, targets=%d", total_interactions, len(target_positions)
    )

    return tensors, target_positions


class DyGKTDataset(Dataset):
    """Simplified dataset for DyGKT - only handles data access."""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        target_indices: np.ndarray,
        num_neighbor: int,
    ):
        self.tensors = tensors
        self.target_indices = target_indices
        self.num_neighbor = num_neighbor

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> int:
        """Return index for collate_fn to batch process."""
        return idx

    def get_batch(
        self, batch_indices: list[int] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(batch_indices):
            batch_indices = torch.tensor(batch_indices, dtype=torch.long)

        data_indices = self.target_indices[batch_indices.numpy()]
        t = self.tensors

        user_his_idx = t["user_his_idx"][data_indices].long()
        que_his_idx = t["que_his_idx"][data_indices].long()

        return {
            "idx": t["idx"][data_indices],
            "user": t["user"][data_indices],
            "question": t["question"][data_indices],
            "skill": t["skill"][data_indices],
            "time": t["time"][data_indices],
            "correctness": t["correctness"][data_indices],
            "src_neighbor_node_ids": t["lookup_question"][user_his_idx],
            "src_neighbor_skill_ids": t["lookup_skill"][user_his_idx],
            "src_neighbor_times": t["lookup_time"][user_his_idx],
            "src_neighbor_edge_feats": t["lookup_correctness"][user_his_idx],
            "src_neighbor_len": t["user_his_len"][data_indices],
            "dst_neighbor_node_ids": t["lookup_user"][que_his_idx],
            "dst_neighbor_times": t["lookup_time"][que_his_idx],
            "dst_neighbor_edge_feats": t["lookup_correctness"][que_his_idx],
            "dst_neighbor_len": t["que_his_len"][data_indices],
        }

    def collate_indices(self, batch_indices: list[int]) -> dict[str, torch.Tensor]:
        return self.get_batch(batch_indices)


class DyGKTModelData(QuestionModelData):
    """Data adapter for DyGKT."""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")

        # Build q_table and config
        q_table = self.build_relationship_matrix(("question", "has", "skill"))
        num_questions = self.data_src.get_metadata("num_questions")
        config = DyGKTNeighborConfig.from_run_config(rc, num_questions)

        # Load sequences
        sequences = self.load_sequences(required=("timestamp", "skills"))
        partition = sequences.partition(rc.data.fold)
        question_sequences = sequences.question
        user_responses = sequences.response
        user_masks = sequences.mask
        user_ids = sequences.user
        # DyGKT uses one skill per event; reserve zero for absent history.
        skill_sequences = to_first_skill_ids(sequences.skills, offset=1)
        time_sequences = sequences.features["timestamp"].astype(np.float64)
        time_sequences = self._normalize_timestamps(time_sequences)

        event_arrays = (
            question_sequences,
            user_responses,
            user_masks,
            time_sequences,
            user_ids,
            skill_sequences,
        )
        train_records = self._build_interaction_records(
            *(array[partition.train] for array in event_arrays)
        )
        val_records = self._build_interaction_records(
            *(array[partition.val] for array in event_arrays)
        )
        test_records = self._build_interaction_records(
            *(array[partition.test] for array in event_arrays)
        )

        # Build datasets
        train_tensors, train_targets = build_tensors(train_records, config)
        train_dataset = DyGKTDataset(train_tensors, train_targets, config.num_neighbor)

        # Val: use train as history
        val_tensors, val_targets = build_tensors(
            val_records,
            config,
            history_data=train_records,
        )
        val_dataset = DyGKTDataset(val_tensors, val_targets, config.num_neighbor)

        # Test: use train+val as history
        test_tensors, test_targets = build_tensors(
            test_records,
            config,
            history_data=train_records + val_records,
        )
        test_dataset = DyGKTDataset(test_tensors, test_targets, config.num_neighbor)

        # Build metadata
        num_users = int(self.data_src.get_metadata("num_users"))

        metadata = {
            "num_questions": num_questions,
            "num_users": num_users,
            "num_skills": int(self.data_src.get_metadata("num_skills")),
            "question_id_offset": 1,
            "user_id_offset": num_questions + 1,
            "question_features": q_table.astype(np.float32),
            "num_neighbor": config.num_neighbor,
        }

        return train_dataset, val_dataset, test_dataset, metadata

    def _normalize_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        ts = np.asarray(timestamps, dtype=np.int64).copy()
        valid = ts != 0
        ts[valid] = ts[valid] // 1000  # 毫秒 → 秒
        return ts

    def _build_interaction_records(
        self,
        question_sequences: np.ndarray,
        user_responses: np.ndarray,
        user_masks: np.ndarray,
        time_sequences: np.ndarray,
        user_ids: np.ndarray,
        skill_sequences: np.ndarray,
    ) -> list[dict[str, Any]]:
        records = []

        for q_seq, r_seq, mask_seq, t_seq, user_id, skill_seq in zip(
            question_sequences,
            user_responses,
            user_masks,
            time_sequences,
            user_ids,
            skill_sequences,
            strict=True,
        ):
            seq_len = int(mask_seq.sum())
            if seq_len <= 0:
                continue

            user_id = int(user_id)

            records.append(
                {
                    "user_id": user_id,
                    "seq_len": seq_len,
                    "question_seq": q_seq[:seq_len].astype(np.int64).tolist(),
                    "skill_seq": skill_seq[:seq_len].astype(np.int64).tolist(),
                    "correctness_seq": r_seq[:seq_len].astype(np.int64).tolist(),
                    "time_seq": t_seq[:seq_len].astype(np.int64).tolist(),
                }
            )

        return records
