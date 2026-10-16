"""Skill and question sequences for ASIKT, with train-fold difficulty priors."""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids


class ASIKTDataset(Dataset):
    def __init__(self, skill, response, mask, question):
        self.skill = skill
        self.response = response
        self.mask = mask
        self.question = question

    def __len__(self):
        return len(self.skill)

    def __getitem__(self, index):
        return (
            torch.as_tensor(self.skill[index], dtype=torch.long),
            torch.as_tensor(self.response[index], dtype=torch.long),
            torch.as_tensor(self.mask[index], dtype=torch.bool),
            torch.as_tensor(self.question[index], dtype=torch.long),
        )


def difficulty_prior(ids, responses, valid, count):
    """Original ASIKT frequency-weighted accuracy prior, fitted on train only."""
    flat_ids = ids[valid].astype(np.int64)
    flat_answers = responses[valid].astype(np.float64)
    frequency = np.bincount(flat_ids, minlength=count).astype(np.float64)
    correct = np.bincount(flat_ids, weights=flat_answers, minlength=count)
    observed = frequency > 0
    if not np.any(observed):
        raise ValueError("ASIKT needs at least one valid training interaction")
    average_frequency = frequency[observed].mean()
    average_accuracy = np.mean(correct[observed] / frequency[observed])
    prior = (correct + average_frequency * average_accuracy) / (
        frequency + average_frequency
    )
    return torch.from_numpy(prior.astype(np.float32)), torch.from_numpy(
        frequency.astype(np.float32)
    )


class ASIKTModelData(SkillModelData):
    def prepare_data(self, rc):
        fold = rc.data.fold
        if fold < 0:
            raise ValueError("ASIKT requires a validation fold")
        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        skill = to_first_skill_ids(sequences.skills)
        response = sequences.response
        mask = sequences.mask
        question = sequences.question
        train, val, _ = partition.apply(ASIKTDataset(skill, response, mask, question))
        n_skill = self.data_src.get_metadata("num_skills")
        n_question = self.data_src.get_metadata("num_questions")
        self.skill_prior, self.skill_frequency = difficulty_prior(
            skill[partition.train],
            response[partition.train],
            mask[partition.train].astype(bool),
            n_skill,
        )
        self.question_prior, self.question_frequency = difficulty_prior(
            question[partition.train],
            response[partition.train],
            mask[partition.train].astype(bool),
            n_question,
        )
        test = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=0,
        )
        return train, val, test
