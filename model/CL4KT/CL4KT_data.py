"""CL4KT data module: sequence augmentation, datasets, and data preparation."""

import math
import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


def _augment_sequence(
    skill_seq: np.ndarray,
    response_seq: np.ndarray,
    mask_seq: np.ndarray,
    seq_len: int,
    mask_token_id: int,
    mask_prob: float,
    crop_prob: float,
    permute_prob: float,
    replace_prob: float,
    negative_prob: float,
    easier_skills: dict,
    harder_skills: dict,
    seed: int,
):
    """Augment one left-aligned sequence for contrastive learning.

    Operates on the de-padded valid prefix, then left-aligns the result back
    to ``seq_len``. Returns ``(aug_skill, aug_response, aug_mask, neg_response)``;
    ``neg_response`` is the response-flip hard-negative view, aligned to the
    original sequence length.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    true_len = int(mask_seq[:seq_len].sum())
    final_neg = np.zeros(seq_len, dtype=np.int64)
    if true_len == 0:
        return (
            np.zeros(seq_len, dtype=np.int64),
            np.zeros(seq_len, dtype=np.int64),
            np.zeros(seq_len, dtype=np.int64),
            final_neg,
        )

    skills = [int(s) for s in skill_seq[:true_len]]
    responses = [int(r) for r in response_seq[:true_len]]

    # BERT-style masking plus per-position response flips for hard negatives.
    aug_s, aug_r, neg_r = [], [], []
    for s, r in zip(skills, responses):
        if mask_prob > 0 and rng.random() < mask_prob:
            sub = rng.random()
            if sub < 0.8:
                aug_s.append(mask_token_id)
            elif sub < 0.9:
                aug_s.append(rng.randint(0, mask_token_id - 1))
            else:
                aug_s.append(s)
        else:
            aug_s.append(s)
        aug_r.append(r)
        if negative_prob > 0 and rng.random() < negative_prob:
            neg_r.append(1 - r)
        else:
            neg_r.append(r)

    # Skill-difficulty replacement: wrong -> harder neighbour, correct -> easier.
    if replace_prob > 0:
        for i, (s, r) in enumerate(zip(aug_s, aug_r)):
            if rng.random() < replace_prob:
                if r == 0 and s in harder_skills:
                    aug_s[i] = harder_skills[s]
                elif r == 1 and s in easier_skills:
                    aug_s[i] = easier_skills[s]

    # Permute a contiguous segment.
    if permute_prob > 0 and len(aug_s) > 1:
        reorder_len = int(math.floor(permute_prob * len(aug_s)))
        if 0 < reorder_len < len(aug_s):
            start = rng.randint(0, len(aug_s) - reorder_len)
            perm = np_rng.permutation(reorder_len)
            seg_s = [aug_s[start + k] for k in perm]
            seg_r = [aug_r[start + k] for k in perm]
            aug_s = aug_s[:start] + seg_s + aug_s[start + reorder_len :]
            aug_r = aug_r[:start] + seg_r + aug_r[start + reorder_len :]

    # Crop a contiguous segment.
    if 0 < crop_prob < 1 and len(aug_s) > 1:
        crop_len = max(1, int(math.floor(crop_prob * len(aug_s))))
        if crop_len < len(aug_s):
            start = rng.randint(0, len(aug_s) - crop_len)
            aug_s = aug_s[start : start + crop_len]
            aug_r = aug_r[start : start + crop_len]

    out_len = len(aug_s)
    final_s = np.zeros(seq_len, dtype=np.int64)
    final_r = np.zeros(seq_len, dtype=np.int64)
    final_m = np.zeros(seq_len, dtype=np.int64)
    final_s[:out_len] = aug_s
    final_r[:out_len] = aug_r
    final_m[:out_len] = 1
    final_neg[: len(neg_r)] = neg_r
    return final_s, final_r, final_m, final_neg


def _compute_skill_neighbours(
    skills: np.ndarray, responses: np.ndarray, masks: np.ndarray
):
    """Map each skill to its easier and harder neighbour by correctness rate."""
    correct = defaultdict(int)
    total = defaultdict(int)
    for s_seq, r_seq, m_seq in zip(skills, responses, masks):
        for s, r, m in zip(s_seq, r_seq, m_seq):
            if m:
                correct[int(s)] += int(r)
                total[int(s)] += 1
    difficulty = {s: correct[s] / total[s] for s in total if total[s] > 0}
    # Hardest (lowest correctness) first.
    ordered = [s for s, _ in sorted(difficulty.items(), key=lambda x: x[1])]
    easier, harder = {}, {}
    n = len(ordered)
    for i, s in enumerate(ordered):
        easier[s] = ordered[i + 1] if i + 1 < n else s
        harder[s] = ordered[i - 1] if i - 1 >= 0 else s
    return easier, harder


class CL4KTTrainDataset(Dataset):
    """Training dataset yielding two augmented views plus the original sequence.

    Each item returns nested tuples for skills, responses, and masks:
    ``(s1, s2, s)``, ``(r1, r2, r, neg_r)``, ``(m1, m2, m)``.
    """

    def __init__(
        self,
        skills: np.ndarray,
        responses: np.ndarray,
        masks: np.ndarray,
        seq_len: int,
        mask_token_id: int,
        aug_probs: dict,
        easier_skills: dict,
        harder_skills: dict,
        sequence_ids: np.ndarray,
    ):
        self.skills = skills.astype(np.int64)
        self.responses = responses.astype(np.int64)
        self.masks = masks.astype(np.int64)
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id
        self.aug_probs = aug_probs
        self.easier_skills = easier_skills
        self.harder_skills = harder_skills
        self.sequence_ids = sequence_ids

    def __len__(self) -> int:
        return len(self.skills)

    def _augment(self, skill_seq, response_seq, mask_seq, seed):
        return _augment_sequence(
            skill_seq,
            response_seq,
            mask_seq,
            self.seq_len,
            self.mask_token_id,
            self.aug_probs["mask_prob"],
            self.aug_probs["crop_prob"],
            self.aug_probs["permute_prob"],
            self.aug_probs["replace_prob"],
            self.aug_probs["negative_prob"],
            self.easier_skills,
            self.harder_skills,
            seed,
        )

    def __getitem__(self, idx: int):
        s = self.skills[idx]
        r = self.responses[idx]
        m = self.masks[idx]
        seed = 2 * int(self.sequence_ids[idx])
        s1, r1, m1, neg_r = self._augment(s, r, m, seed=seed)
        s2, r2, m2, _ = self._augment(s, r, m, seed=seed + 1)
        skills = (
            torch.from_numpy(s1),
            torch.from_numpy(s2),
            torch.from_numpy(s.astype(np.int64)),
        )
        responses = (
            torch.from_numpy(r1),
            torch.from_numpy(r2),
            torch.from_numpy(r.astype(np.int64)),
            torch.from_numpy(neg_r),
        )
        masks = (
            torch.from_numpy(m1),
            torch.from_numpy(m2),
            torch.from_numpy(m.astype(np.int64)),
        )
        return skills, responses, masks


class CL4KTEvalDataset(Dataset):
    """Evaluation dataset yielding the original ``(skill, response, mask)``."""

    def __init__(self, skills: np.ndarray, responses: np.ndarray, masks: np.ndarray):
        self.skills = torch.from_numpy(skills.astype(np.int64))
        self.responses = torch.from_numpy(responses.astype(np.int64))
        self.masks = torch.from_numpy(masks.astype(np.int64))

    def __len__(self) -> int:
        return len(self.skills)

    def __getitem__(self, idx: int):
        return self.skills[idx], self.responses[idx], self.masks[idx]


class CL4KTModelData(SkillModelData):
    """Prepares augmented training data and windowlate test data for CL4KT."""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = to_first_skill_ids(sequences.skills)
        user_response = sequences.response
        user_mask = sequences.mask

        train_s = user_sequence[partition.train]
        train_r = user_response[partition.train]
        train_m = user_mask[partition.train]

        seq_len = self.data_src.get_metadata("max_skill_seq_len")
        num_skills = self.data_src.get_metadata("num_skills")
        mask_token_id = num_skills

        m = rc.model
        aug_probs = {
            "mask_prob": m.mask_prob,
            "crop_prob": m.crop_prob,
            "permute_prob": m.permute_prob,
            "replace_prob": m.replace_prob,
            "negative_prob": m.negative_prob,
        }
        easier_skills, harder_skills = _compute_skill_neighbours(
            train_s, train_r, train_m
        )

        train_dataset = CL4KTTrainDataset(
            user_sequence,
            user_response,
            user_mask,
            seq_len,
            mask_token_id,
            aug_probs,
            easier_skills,
            harder_skills,
            sequences.sequence_id,
        )
        train_dataset = Subset(train_dataset, partition.train.tolist())
        val_dataset = Subset(
            CL4KTEvalDataset(user_sequence, user_response, user_mask),
            partition.val.tolist(),
        )
        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"CL4KT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset
