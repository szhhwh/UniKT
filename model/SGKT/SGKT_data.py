"""
SGKT data preparation module.

Implements data preparation for Session Graph-based Knowledge Tracing model.
Builds the Heterogeneous Relation Graph (HRG) following the original author's strategy.
"""

from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


def sample_hist_neighbors(
    batch_size,
    max_seq_len,
    hist_neighbor_num,
    skill_index,
    pad_index=None,
    deterministic=False,
):
    """
    Sample historical neighbors based on skill matching.

    Following the original TF implementation (data_process.py:198-220).
    For each position t, find historical positions [0, t-1] with the same skill,
    then randomly select M positions as neighbors.

    Args:
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        hist_neighbor_num: Number of historical neighbors to sample (M)
        skill_index: [batch_size, max_seq_len] Skill indices (tensor or ndarray)
        pad_index: Padding index used when no valid historical neighbor exists.
                   Defaults to max_seq_len and requires downstream gather to
                   append one zero-padding position.

    Returns:
        hist_neighbor_index: [batch_size, max_seq_len, hist_neighbor_num]
                            Pre-computed historical neighbor indices
    """
    if pad_index is None:
        pad_index = max_seq_len

    if hist_neighbor_num == 0:
        return np.zeros((batch_size, max_seq_len, 0), dtype=np.int64)

    # Convert to numpy if tensor
    if isinstance(skill_index, torch.Tensor):
        skills = skill_index.numpy()
    else:
        skills = np.asarray(skill_index)

    # Result array: default to pad_index (dedicated padding position for no-match fallback)
    result = np.full(
        (batch_size, max_seq_len, hist_neighbor_num),
        pad_index,
        dtype=np.int64,
    )
    # Position 0 has no history; keep padding index to avoid future leakage.
    result[:, 0, :] = pad_index

    for b in range(batch_size):
        seq_skills = skills[b]  # [max_seq_len]

        # Build same-skill match matrix: same_skill[t, k] = True if skills match
        # and k < t (causal: only look at history)
        # [max_seq_len] == [max_seq_len, 1] -> [max_seq_len, max_seq_len]
        same_skill = seq_skills[np.newaxis, :] == seq_skills[:, np.newaxis]
        # Causal mask: position k < t
        causal = np.tril(np.ones((max_seq_len, max_seq_len), dtype=bool), k=-1)
        valid = same_skill & causal  # [max_seq_len, max_seq_len]

        for t in range(1, max_seq_len):
            candidates = np.where(valid[t])[0]
            n_candidates = len(candidates)
            if deterministic and n_candidates > 0:
                # Fixed earliest-M pick keeps eval reproducible.
                result[b, t] = candidates[np.arange(hist_neighbor_num) % n_candidates]
            elif n_candidates >= hist_neighbor_num:
                result[b, t] = np.random.choice(
                    candidates, hist_neighbor_num, replace=False
                )
            elif n_candidates > 0:
                result[b, t] = np.random.choice(
                    candidates, hist_neighbor_num, replace=True
                )
            # else: keeps default pad_index

    return result


class SGKTModelData(QuestionModelData):
    """
    SGKT data preparation class.

    Builds:
    1. User sequence data
    2. Question-skill relationship matrix
    3. HRG (Heterogeneous Relation Graph) with GCNConv-compatible format
    """

    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, rc):
        """
        Prepare data for SGKT model.

        Args:
            rc: RunConfig (OmegaConf DictConfig)

        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
            hrg_data: PyG Data object for HRG graph
            question_skill_matrix: Question-skill relationship matrix
            num_skills: Number of skills
            num_questions: Number of questions
        """
        # 1. Build sequence data
        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        np.where(sequences.mask, sequences.user[:, None], 0)

        # 2. Build question-skill relationship matrix
        question_skill_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )

        # Get metadata
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        logger.info("Building question-skill neighbors for HRG")

        train_user_indices = partition.train
        question_neighbors, skill_neighbors = self.build_qs_neighbors(
            question_skill_matrix=question_skill_matrix,
            user_sequence=user_sequence[train_user_indices],
            user_mask=user_mask[train_user_indices],
            num_skills=num_skills,
            num_questions=num_questions,
            question_neighbor_num=rc.model.question_neighbor_num,
            skill_neighbor_num=rc.model.skill_neighbor_num,
        )
        question_neighbors[:num_skills] = skill_neighbors

        # 4. Split data into train/val or k-fold
        user_skills = to_first_skill_ids(sequences.skills)

        # Unpack train/val/test data

        hist_neighbor_num = rc.model.hist_neighbor_num

        dataset = SGKTDataset(
            user_sequence, user_response, user_mask, user_skills, hist_neighbor_num
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        # Create custom collate function with hist_neighbor_num
        train_collate_fn = partial(sgkt_collate_fn, hist_neighbor_num=hist_neighbor_num)
        # Deterministic eval collate keeps val/test metrics reproducible.
        val_collate_fn = partial(
            sgkt_collate_fn, hist_neighbor_num=hist_neighbor_num, deterministic=True
        )

        logger.info(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
        )

        hrg_context = {
            "question_neighbors": torch.from_numpy(question_neighbors).long(),
            "feature_embedding": None,
            "next_neighbor_num": rc.model.next_neighbor_num,
        }

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            hrg_context,
            num_skills,
            num_questions,
            train_collate_fn,
            val_collate_fn,
        )

    def build_qs_neighbors(
        self,
        question_skill_matrix,
        user_sequence,
        user_mask,
        num_skills,
        num_questions,
        question_neighbor_num,
        skill_neighbor_num,
    ):
        qs_num = num_skills + num_questions
        # Use set-backed adjacency for O(1) deduplication while preserving
        # the original graph semantics.
        adj_sets = [set() for _ in range(qs_num)]

        question_skills = [
            np.where(question_skill_matrix[q_id] == 1)[0].tolist()
            for q_id in range(num_questions)
        ]

        # 1) Filter out padded positions using mask (vectorized).
        valid_positions = user_mask.astype(bool)
        valid_questions = user_sequence[valid_positions].astype(np.int64, copy=False)
        valid_questions = valid_questions[
            (valid_questions >= 0) & (valid_questions < num_questions)
        ]

        # 2) Build question<->skill edges for questions appearing in the given sequences.
        if valid_questions.size > 0:
            appeared_questions = np.unique(valid_questions)
            for q_id in appeared_questions.tolist():
                q_node = num_skills + q_id
                skill_neighbors_for_q = question_skills[q_id]
                adj_sets[q_node].update(skill_neighbors_for_q)
                for skill_id in skill_neighbors_for_q:
                    adj_sets[skill_id].add(q_node)

        # Add question-question co-occurrence edges within each sequence (real interactions only).
        for seq, mask in zip(user_sequence, user_mask):
            valid_seq = seq[mask.astype(bool)]
            if valid_seq.size == 0:
                continue

            valid_seq = valid_seq[(valid_seq >= 0) & (valid_seq < num_questions)]
            if valid_seq.size == 0:
                continue

            # Keep first-occurrence order in each sequence (same as previous behavior).
            unique_q, first_idx = np.unique(valid_seq, return_index=True)
            ordered_q = unique_q[np.argsort(first_idx)]
            question_nodes = (num_skills + ordered_q).tolist()

            if len(question_nodes) <= 1:
                continue

            # Add complete co-occurrence edges among questions in the same sequence.
            question_node_set = set(question_nodes)
            for q_node in question_nodes:
                adj_sets[q_node].update(question_node_set - {q_node})

        # Convert set-backed adjacency into lists for downstream sampling.
        adj_list = [list(neighbors) for neighbors in adj_sets]

        question_neighbors = np.zeros([qs_num, question_neighbor_num], dtype=np.int32)
        skill_neighbors = np.zeros([num_skills, skill_neighbor_num], dtype=np.int32)
        for index, neighbors in enumerate(adj_list):
            if index < num_skills:
                if len(neighbors) > 0:
                    if len(neighbors) >= skill_neighbor_num:
                        skill_neighbors[index] = np.random.choice(
                            neighbors, skill_neighbor_num, replace=False
                        )
                    else:
                        skill_neighbors[index] = np.random.choice(
                            neighbors, skill_neighbor_num, replace=True
                        )
            else:
                if len(neighbors) > 0:
                    if len(neighbors) >= question_neighbor_num:
                        neighbors_arr = np.asarray(neighbors, dtype=np.int32)
                        save_skill = neighbors_arr[neighbors_arr < num_skills]
                        save_question = neighbors_arr[neighbors_arr >= num_skills]
                        if len(save_skill) >= question_neighbor_num:
                            question_neighbors[index] = np.random.choice(
                                save_skill, question_neighbor_num, replace=False
                            )
                        else:
                            question_neighbors[index][: len(save_skill)] = save_skill
                            if question_neighbor_num - len(save_skill) - 1 > 0:
                                temp = np.random.choice(
                                    save_question,
                                    question_neighbor_num - len(save_skill) - 1,
                                    replace=False,
                                )
                                changdu = len(save_skill)
                                question_neighbors[
                                    index, changdu + 1 : changdu + 1 + len(temp)
                                ] = temp
                    else:
                        question_neighbors[index] = np.random.choice(
                            neighbors, question_neighbor_num, replace=True
                        )

        return question_neighbors, skill_neighbors


class SGKTDataset(Dataset):
    """
    Dataset for SGKT model.

    Args:
        user_sequence: [num_users, max_seq_len] Question IDs
        user_response: [num_users, max_seq_len] Responses (0/1)
        user_mask: [num_users, max_seq_len] Valid position mask
        user_skills: [num_users, max_seq_len] Skill IDs
        hist_neighbor_num: Number of historical neighbors (M)
    """

    def __init__(
        self, user_sequence, user_response, user_mask, user_skills, hist_neighbor_num
    ):
        self.user_sequence = torch.from_numpy(user_sequence).long()
        self.user_response = torch.from_numpy(user_response).long()
        self.user_mask = torch.from_numpy(user_mask).bool()
        self.user_skills = torch.from_numpy(user_skills).long()
        self.hist_neighbor_num = hist_neighbor_num
        self.max_seq_len = user_sequence.shape[1]

        assert (
            len(user_sequence)
            == len(user_response)
            == len(user_mask)
            == len(user_skills)
        ), "Sequence, response, mask, and skills must have the same length"

    def __len__(self):
        return len(self.user_sequence)

    def __getitem__(self, idx):
        return {
            "sequence": self.user_sequence[idx],
            "response": self.user_response[idx],
            "mask": self.user_mask[idx],
            "skills": self.user_skills[idx],
        }


def sgkt_collate_fn(batch, hist_neighbor_num=5, deterministic=False):
    """
    Custom collate function for SGKT that computes hist_neighbor_index per batch.

    Args:
        batch: List of dicts with keys: sequence, response, mask, skills
        hist_neighbor_num: Number of historical neighbors (M)

    Returns:
        Batched dict with hist_neighbor_index added
    """
    from torch.utils.data.dataloader import default_collate

    # First, batch the regular tensors
    batched = default_collate(batch)

    # Compute hist_neighbor_index for this batch
    batch_size = batched["skills"].shape[0]
    full_seq_len = batched["skills"].shape[1]  # e.g., 200
    model_seq_len = full_seq_len - 1  # e.g., 199

    # Compute hist_neighbor_index using model sequence length
    hist_neighbor_index = sample_hist_neighbors(
        batch_size,
        model_seq_len,  # Use model_seq_len instead of full_seq_len
        hist_neighbor_num,
        batched["skills"][:, :model_seq_len],  # Slice skills to model_seq_len
        pad_index=model_seq_len,
        deterministic=deterministic,
    )

    batched["hist_neighbor_index"] = torch.from_numpy(hist_neighbor_index).long()

    return batched
