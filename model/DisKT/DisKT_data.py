"""DisKT data preparation: question-level sequences with collapsed concepts."""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_skill_combination_ids

logger = get_logger(__name__)


class DisKTDataset(Dataset):
    """Per-sample question / concept / response / masks / counterfactual flags."""

    def __init__(self, questions, skills, responses, masks, counter_masks):
        self.questions = torch.from_numpy(np.asarray(questions)).long()
        self.skills = torch.from_numpy(np.asarray(skills)).long()
        self.responses = torch.from_numpy(np.asarray(responses)).long()
        self.masks = torch.from_numpy(np.asarray(masks)).bool()
        self.counter_masks = torch.from_numpy(np.asarray(counter_masks)).long()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "questions": self.questions[idx],
            "skills": self.skills[idx],
            "responses": self.responses[idx],
            "masks": self.masks[idx],
            "counter_masks": self.counter_masks[idx],
        }


class DisKTModelData(QuestionModelData):
    """Question-level data with each interaction's skills folded to one concept.

    Each interaction provides its skill set; distinct sets each map to
    one new concept, so every interaction has exactly one concept id. Concept ids
    are 1-indexed (0 reserved for padding).
    """

    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, rc):
        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_question = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        num_questions = self.data_src.get_metadata("num_questions")

        # Multi-skill folding: collapse shared skill sets into single concepts.
        skills_sequence, num_concepts = to_skill_combination_ids(sequences.skills)
        num_skills = num_concepts + 1

        skill_difficulty = self._compute_skill_difficulty(
            skills_sequence[partition.train],
            user_response[partition.train],
            user_mask[partition.train],
            num_skills,
        )
        counters = np.empty_like(user_response, dtype=np.int8)
        for seed, indices in enumerate(
            (partition.train, partition.val, partition.test)
        ):
            counters[indices] = self._build_counter_masks(
                skills_sequence[indices],
                user_response[indices],
                user_mask[indices],
                skill_difficulty,
                seed_base=seed,
            )
        dataset = DisKTDataset(
            user_question, skills_sequence, user_response, user_mask, counters
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset, num_skills, num_questions

    @staticmethod
    def _compute_skill_difficulty(skills, responses, masks, num_skills):
        """Per-concept mean correctness over valid training interactions."""
        skills = np.asarray(skills)
        responses = np.asarray(responses)
        masks = np.asarray(masks)

        valid = masks > 0
        flat_s = skills[valid]
        flat_r = responses[valid].astype(np.float64)

        sums = np.bincount(flat_s, weights=flat_r, minlength=num_skills)
        counts = np.bincount(flat_s, minlength=num_skills)
        with np.errstate(divide="ignore", invalid="ignore"):
            difficulty = np.where(counts > 0, sums / np.maximum(counts, 1), 0.5)
        return difficulty

    @staticmethod
    def _build_counter_masks(skills, responses, masks, skill_difficulty, seed_base):
        """Vectorised counterfactual mask.

        A position is flagged when its counterfactual score falls below the
        running mean of preceding valid positions. Mirrors the online
        ``alpha_double`` threshold in the reference implementation.
        """
        skills = np.asarray(skills)
        responses = np.asarray(responses)
        masks = np.asarray(masks)

        n, seq_len = skills.shape
        counter_masks = np.zeros((n, seq_len), dtype=np.int8)

        valid_all = skills != 0
        diff = skill_difficulty[skills]

        rng = np.random.RandomState(seed_base)
        for i in range(n):
            r = responses[i].astype(np.float64)
            probs = rng.random_sample(seq_len)
            contr = np.maximum(probs, 0.1) * (1.0 - r + (2.0 * r - 1.0) * diff[i])

            valid = valid_all[i]
            contr_valid = np.where(valid, contr, 0.0)
            cumsum = np.cumsum(contr_valid)
            cumcnt = np.cumsum(valid.astype(np.int32))
            alpha_after = np.where(cumcnt > 0, cumsum / np.maximum(cumcnt, 1), 0.0)
            # Initial threshold 0.2 at the first valid position; running mean of prior valid scores thereafter.
            alpha_before = np.full(seq_len, 0.2, dtype=np.float64)
            alpha_before[1:] = np.where(cumcnt[:-1] > 0, alpha_after[:-1], 0.2)

            counter_masks[i] = ((contr < alpha_before) & valid).astype(np.int8)

        return counter_masks
